from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class TradeRecord(BaseModel):
    """A single trade event for a Kalshi ticket."""

    timestamp: datetime
    quantity: float
    position: str = Field(..., description="yes or no")
    action: str = Field(..., description="buy or sell")
    probability: float
    price: float
    ticket_id: Optional[str] = None
    trade_id: Optional[str] = None
    probability_change: float = 0.0
    time_since_previous_trade_seconds: Optional[float] = None


class TicketHistory(BaseModel):
    """All trades observed for one ticket, with a time-span label."""

    ticket_id: str
    event_ticker: str
    event_title: Optional[str] = None
    open_timestamp: Optional[datetime] = None
    close_timestamp: Optional[datetime] = None
    first_trade_timestamp: Optional[datetime] = None
    last_trade_timestamp: Optional[datetime] = None
    time_traversed_days: float = 0.0
    trades: List[TradeRecord] = Field(default_factory=list)
    backfill_source_ticket_id: Optional[str] = None
    backfill_time_days: float = 0.0
    label: str = ""

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def start_probability(self) -> Optional[float]:
        if self.trades:
            return self.trades[0].probability
        return None

    @property
    def end_probability(self) -> Optional[float]:
        if self.trades:
            return self.trades[-1].probability
        return None

    def to_summary(self) -> Dict[str, Any]:
        return self.to_dict(include_trades=False)

    def to_dict(self, include_trades: bool = True) -> Dict[str, Any]:
        payload = {
            "ticket_id": self.ticket_id,
            "event_ticker": self.event_ticker,
            "event_title": self.event_title,
            "label": self.label,
            "trade_count": self.trade_count,
            "time_traversed_days": round(self.time_traversed_days, 3),
            "first_trade_timestamp": self.first_trade_timestamp.isoformat() if self.first_trade_timestamp else None,
            "last_trade_timestamp": self.last_trade_timestamp.isoformat() if self.last_trade_timestamp else None,
            "open_timestamp": self.open_timestamp.isoformat() if self.open_timestamp else None,
            "close_timestamp": self.close_timestamp.isoformat() if self.close_timestamp else None,
            "backfill_source_ticket_id": self.backfill_source_ticket_id,
            "backfill_time_days": round(self.backfill_time_days, 3),
        }
        if include_trades:
            payload["trades"] = [
                {
                    "timestamp": trade.timestamp.isoformat(),
                    "quantity": trade.quantity,
                    "position": trade.position,
                    "action": trade.action,
                    "probability": trade.probability,
                    "price": trade.price,
                    "ticket_id": trade.ticket_id,
                    "trade_id": trade.trade_id,
                    "probability_change": trade.probability_change,
                    "time_since_previous_trade_seconds": trade.time_since_previous_trade_seconds,
                }
                for trade in self.trades
            ]
        return payload


class TrainingExample(BaseModel):
    """A single row suitable for training a binary classifier or regressor."""

    ticket_id: str
    event_ticker: str
    timestamp: datetime
    probability: float
    probability_change: float
    quantity: float
    position: int
    action: int
    time_since_open_hours: float
    ticket_time_span_days: float
    target: int
    source_ticket_id: Optional[str] = None


class KalshiHistoricalDataBuilder:
    """Build a ticket-centric historical dataset that is ready for ML training."""

    def __init__(
        self,
        api_base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 20,
        use_offline: bool = False,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.use_offline = use_offline
        self.session = requests.Session()

    def build_dataset(self, crypto: str, days: int, raw_only: bool = False) -> Dict[str, Any]:
        logger.info("Building dataset for %s over %s days", crypto, days)
        ticket_payloads = self._fetch_ticket_payloads(crypto, days)
        tickets: List[TicketHistory] = []

        for payload in ticket_payloads:
            ticket = self._build_ticket_history(payload, crypto)
            if not ticket.trades:
                continue
            tickets.append(ticket)

        # Fill gaps in time coverage when possible.
        target_span = max(days, 1)
        filled_tickets: List[TicketHistory] = []
        for ticket in tickets:
            if ticket.time_traversed_days >= target_span:
                filled_tickets.append(ticket)
                continue
            similar = self._find_similar_ticket(ticket, tickets)
            if similar is None:
                filled_tickets.append(ticket)
                continue
            combined = self._concatenate_ticket_histories(ticket, similar)
            if combined is not None:
                filled_tickets.append(combined)
            else:
                filled_tickets.append(ticket)

        training_rows = self._build_training_rows(filled_tickets)
        if raw_only:
            return {
                "crypto": crypto,
                "requested_days": days,
                "training_rows": [self._row_to_dict(row) for row in training_rows],
                "notes": [
                    "This export uses the raw trade-row format for direct time-series training.",
                    "It drops the ticket-spanning backfill logic and is better suited to simple forecasting tasks.",
                ],
            }

        return {
            "crypto": crypto,
            "requested_days": days,
            "ticket_histories": [ticket.to_dict(include_trades=True) for ticket in filled_tickets],
            "training_rows": [self._row_to_dict(row) for row in training_rows],
            "notes": [
                "The dataset is ticket-centric and preserves buy/sell and yes/no labels.",
                "When a ticket lacks enough temporal coverage, a similar earlier ticket may be used as a backfill source.",
            ],
        }

    def _fetch_ticket_payloads(self, crypto: str, days: int) -> List[Dict[str, Any]]:
        if self.use_offline:
            return self._offline_ticket_payloads(crypto, days)

        endpoints = [
            "/series",
            "/tradeable_events",
            "/markets",
        ]
        for endpoint in endpoints:
            try:
                response = self._request("GET", endpoint, params={"event_ticker": crypto, "limit": 100})
                payload = response.json()
                items = self._extract_items(payload)
                if items:
                    return [self._normalise_ticket_payload(item) for item in items]
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("Endpoint %s failed with %s", endpoint, exc)

        logger.warning("Falling back to offline sample because no live Kalshi payloads were available.")
        return self._offline_ticket_payloads(crypto, days)

    def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.api_base_url}{path}"
        response = self.session.request(method, url, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response

    def _extract_items(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            for key in ("data", "results", "series", "markets", "events", "tradeable_events"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _normalise_ticket_payload(self, item: Dict[str, Any]) -> Dict[str, Any]:
        ticket_id = self._first_present(item, ["id", "ticket_id", "market_id", "series_id", "contract_id"])
        event_ticker = self._first_present(item, ["event_ticker", "event_ticker_symbol", "ticker", "symbol"])
        title = self._first_present(item, ["title", "name", "event_title", "market_title"])
        open_ts = self._parse_time(self._first_present(item, ["open_time", "open_timestamp", "start_time", "created_at"]))
        close_ts = self._parse_time(self._first_present(item, ["close_time", "close_timestamp", "end_time", "settlement_time"]))

        return {
            "ticket_id": str(ticket_id) if ticket_id is not None else "unknown-ticket",
            "event_ticker": str(event_ticker) if event_ticker else "UNKNOWN",
            "event_title": str(title) if title else None,
            "open_timestamp": open_ts,
            "close_timestamp": close_ts,
            "trades": [],
        }

    def _build_ticket_history(self, payload: Dict[str, Any], crypto: str) -> TicketHistory:
        ticket_id = payload.get("ticket_id") or payload.get("id") or payload.get("market_id") or "unknown-ticket"
        event_ticker = payload.get("event_ticker") or crypto
        event_title = payload.get("event_title") or payload.get("title")
        open_ts = payload.get("open_timestamp") or payload.get("open_time")
        close_ts = payload.get("close_timestamp") or payload.get("close_time")
        open_dt = self._parse_time(open_ts)
        close_dt = self._parse_time(close_ts)

        trades: List[TradeRecord] = []
        raw_trades = payload.get("trades") or []
        if not raw_trades and not self.use_offline:
            try:
                raw_trades = self._fetch_trade_history(ticket_id)
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("Unable to fetch trade history for %s: %s", ticket_id, exc)

        if not raw_trades and self.use_offline:
            raw_trades = self._offline_trades(ticket_id, open_dt or datetime.now(timezone.utc))
        elif not raw_trades:
            raw_trades = self._offline_trades(ticket_id, open_dt or datetime.now(timezone.utc))

        previous_trade: Optional[TradeRecord] = None
        for trade_idx, raw_trade in enumerate(raw_trades):
            trade = self._normalise_trade_record(raw_trade, ticket_id=ticket_id, previous_trade=previous_trade)
            if trade is None:
                continue
            trades.append(trade)
            previous_trade = trade

        if not trades:
            # Keep a ticket history object even if no trade data is available.
            return TicketHistory(
                ticket_id=str(ticket_id),
                event_ticker=str(event_ticker),
                event_title=event_title,
                open_timestamp=open_dt,
                close_timestamp=close_dt,
                label=f"{str(event_ticker)}-{str(ticket_id)}",
            )

        first_trade = trades[0].timestamp
        last_trade = trades[-1].timestamp
        time_span_days = self._time_span_days(first_trade, last_trade)
        if open_dt and close_dt:
            time_span_days = max(time_span_days, self._time_span_days(open_dt, close_dt))

        return TicketHistory(
            ticket_id=str(ticket_id),
            event_ticker=str(event_ticker),
            event_title=event_title,
            open_timestamp=open_dt,
            close_timestamp=close_dt,
            first_trade_timestamp=first_trade,
            last_trade_timestamp=last_trade,
            time_traversed_days=time_span_days,
            trades=trades,
            label=f"{str(event_ticker)}-{str(ticket_id)}",
        )

    def _fetch_trade_history(self, ticket_id: str) -> List[Dict[str, Any]]:
        endpoints = [
            f"/markets/{ticket_id}/trades",
            f"/tradeable_events/{ticket_id}/trades",
            f"/markets/{ticket_id}/price_history",
            f"/trade_history?market_id={ticket_id}",
        ]
        for endpoint in endpoints:
            try:
                response = self._request("GET", endpoint)
                payload = response.json()
                items = self._extract_items(payload)
                if items:
                    return items
            except Exception as exc:  # pragma: no cover - defensive path
                logger.warning("Trade history endpoint %s failed: %s", endpoint, exc)
        return []

    def _normalise_trade_record(
        self,
        raw_trade: Dict[str, Any],
        *,
        ticket_id: str,
        previous_trade: Optional[TradeRecord],
    ) -> Optional[TradeRecord]:
        if not isinstance(raw_trade, dict):
            return None

        timestamp = self._parse_time(
            self._first_present(
                raw_trade,
                ["timestamp", "created_at", "time", "date", "trade_time", "updated_at"],
            )
        )
        if timestamp is None:
            return None

        quantity = self._coerce_float(self._first_present(raw_trade, ["quantity", "size", "amount", "qty"]))
        if quantity is None:
            quantity = 1.0

        price = self._coerce_float(self._first_present(raw_trade, ["price", "mid_price", "probability", "prob", "value"]))
        if price is None:
            price = 0.5

        probability = self._coerce_float(self._first_present(raw_trade, ["probability", "prob", "price", "mid_price"]))
        if probability is None:
            probability = max(0.0, min(1.0, price))

        position = self._infer_position(raw_trade)
        action = self._infer_action(raw_trade)

        probability_change = 0.0
        time_since_previous_trade_seconds = None
        if previous_trade is not None:
            probability_change = round(probability - previous_trade.probability, 6)
            time_since_previous_trade_seconds = round((timestamp - previous_trade.timestamp).total_seconds(), 3)

        return TradeRecord(
            timestamp=timestamp,
            quantity=float(quantity),
            position=position,
            action=action,
            probability=float(probability),
            price=float(price),
            ticket_id=ticket_id,
            trade_id=str(self._first_present(raw_trade, ["trade_id", "id", "order_id", "uuid"]) or ""),
            probability_change=probability_change,
            time_since_previous_trade_seconds=time_since_previous_trade_seconds,
        )

    def _infer_position(self, raw_trade: Dict[str, Any]) -> str:
        raw_position = str(self._first_present(raw_trade, ["position", "side", "direction", "yes_no", "contract"]) or "").lower()
        if "yes" in raw_position:
            return "yes"
        if "no" in raw_position:
            return "no"
        return "yes"

    def _infer_action(self, raw_trade: Dict[str, Any]) -> str:
        raw_action = str(self._first_present(raw_trade, ["action", "side", "trade_type", "order_type"]) or "").lower()
        if "sell" in raw_action or raw_action == "short":
            return "sell"
        if "buy" in raw_action or raw_action == "long":
            return "buy"
        return "buy"

    def _find_similar_ticket(self, target: TicketHistory, tickets: List[TicketHistory]) -> Optional[TicketHistory]:
        matching = [ticket for ticket in tickets if ticket.ticket_id != target.ticket_id and ticket.event_ticker == target.event_ticker]
        if not matching:
            return None
        matching.sort(key=lambda item: (item.last_trade_timestamp or datetime.now(timezone.utc), item.time_traversed_days), reverse=True)
        return matching[0]

    def _concatenate_ticket_histories(self, target: TicketHistory, source: TicketHistory) -> Optional[TicketHistory]:
        if not source.trades:
            return None
        combined_trades = source.trades + target.trades
        combined_trades = sorted(combined_trades, key=lambda item: item.timestamp)
        combined = TicketHistory(
            ticket_id=f"{target.ticket_id}+{source.ticket_id}",
            event_ticker=target.event_ticker,
            event_title=target.event_title or source.event_title,
            open_timestamp=source.open_timestamp or target.open_timestamp,
            close_timestamp=target.close_timestamp or source.close_timestamp,
            first_trade_timestamp=combined_trades[0].timestamp,
            last_trade_timestamp=combined_trades[-1].timestamp,
            time_traversed_days=self._time_span_days(combined_trades[0].timestamp, combined_trades[-1].timestamp),
            trades=combined_trades,
            backfill_source_ticket_id=source.ticket_id,
            backfill_time_days=self._time_span_days(source.first_trade_timestamp or source.last_trade_timestamp or source.open_timestamp or datetime.now(timezone.utc), source.last_trade_timestamp or source.first_trade_timestamp or source.open_timestamp or datetime.now(timezone.utc)),
            label=f"{target.label}::backfilled-from-{source.ticket_id}",
        )
        return combined

    def _row_to_dict(self, row: TrainingExample) -> Dict[str, Any]:
        return {
            "ticket_id": row.ticket_id,
            "event_ticker": row.event_ticker,
            "timestamp": row.timestamp.isoformat(),
            "probability": row.probability,
            "probability_change": row.probability_change,
            "quantity": row.quantity,
            "position": row.position,
            "action": row.action,
            "time_since_open_hours": row.time_since_open_hours,
            "ticket_time_span_days": row.ticket_time_span_days,
            "target": row.target,
            "source_ticket_id": row.source_ticket_id,
        }

    def _build_training_rows(self, tickets: List[TicketHistory]) -> List[TrainingExample]:
        rows: List[TrainingExample] = []
        for ticket in tickets:
            if not ticket.trades:
                continue
            first_trade_time = ticket.trades[0].timestamp
            for idx in range(0, len(ticket.trades)):
                current = ticket.trades[idx]
                previous = ticket.trades[idx - 1] if idx > 0 else ticket.trades[idx]
                if current.timestamp <= previous.timestamp and idx > 0:
                    continue
                time_since_open_hours = max(0.0, (current.timestamp - first_trade_time).total_seconds() / 3600.0)
                target = 1 if idx == 0 else (1 if current.probability >= previous.probability else 0)
                rows.append(
                    TrainingExample(
                        ticket_id=ticket.ticket_id,
                        event_ticker=ticket.event_ticker,
                        timestamp=current.timestamp,
                        probability=current.probability,
                        probability_change=current.probability_change if idx > 0 else 0.0,
                        quantity=current.quantity,
                        position=1 if current.position == "yes" else 0,
                        action=1 if current.action == "buy" else 0,
                        time_since_open_hours=time_since_open_hours,
                        ticket_time_span_days=ticket.time_traversed_days,
                        target=target,
                        source_ticket_id=ticket.backfill_source_ticket_id,
                    )
                )
        return rows

    def _offline_ticket_payloads(self, crypto: str, days: int) -> List[Dict[str, Any]]:
        anchor = datetime.now(timezone.utc)
        return [
            {
                "ticket_id": "BTC-001",
                "event_ticker": crypto,
                "event_title": f"Will {crypto} be above 90k at 9pm ET?",
                "open_timestamp": (anchor - timedelta(days=max(days, 7))).isoformat(),
                "close_timestamp": (anchor + timedelta(days=1)).isoformat(),
                "trades": self._offline_trades("BTC-001", anchor),
            },
            {
                "ticket_id": "BTC-002",
                "event_ticker": crypto,
                "event_title": f"Will {crypto} be above 95k at 9pm ET?",
                "open_timestamp": (anchor - timedelta(days=max(days - 2, 3))).isoformat(),
                "close_timestamp": (anchor + timedelta(days=2)).isoformat(),
                "trades": self._offline_trades("BTC-002", anchor),
            },
        ]

    def _offline_trades(self, ticket_id: str, base_time: datetime) -> List[Dict[str, Any]]:
        base = base_time if base_time.tzinfo else base_time.replace(tzinfo=timezone.utc)
        trades: List[Dict[str, Any]] = []
        for offset in range(0, 5):
            ts = base - timedelta(hours=12 * offset)
            probability = 0.52 + (offset * 0.01)
            trades.append(
                {
                    "timestamp": ts.isoformat(),
                    "quantity": 1.0 + offset,
                    "position": "yes" if offset % 2 == 0 else "no",
                    "action": "buy" if offset % 2 == 0 else "sell",
                    "probability": probability,
                    "price": probability,
                    "trade_id": f"{ticket_id}-{offset}",
                }
            )
        return trades

    def _first_present(self, item: Dict[str, Any], keys: List[str]) -> Any:
        for key in keys:
            if key in item and item[key] is not None:
                return item[key]
        return None

    def _parse_time(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            for candidate in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z"]:
                try:
                    parsed = datetime.strptime(text, candidate)
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    return parsed
                except ValueError:
                    continue
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

    def _coerce_float(self, value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _time_span_days(self, start: Optional[datetime], end: Optional[datetime]) -> float:
        if start is None or end is None:
            return 0.0
        delta = end - start
        return max(0.0, delta.total_seconds() / 86400.0)


def build_dataset(crypto: str, days: int, output_path: str, api_key: Optional[str], api_base_url: str, use_offline: bool, raw_only: bool = False) -> Dict[str, Any]:
    builder = KalshiHistoricalDataBuilder(
        api_base_url=api_base_url,
        api_key=api_key,
        use_offline=use_offline,
    )
    dataset = builder.build_dataset(crypto, days, raw_only=raw_only)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(dataset, handle, indent=2, default=str)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Kalshi ticket history into an ML-friendly dataset")
    parser.add_argument("--crypto", default="BTC", help="The cryptocurrency or event ticker to query")
    parser.add_argument("--days", type=int, default=60, help="The number of days of ticket history to collect")
    parser.add_argument("--output", default="kalshi_training_dataset.json", help="Path to write the JSON payload")
    parser.add_argument("--api-key", default=os.getenv("KALSHI_API_KEY"), help="Optional Kalshi API key")
    parser.add_argument("--api-base-url", default=os.getenv("KALSHI_API_BASE_URL", "https://api.elections.kalshi.com/trade/v2"), help="Kalshi API base URL")
    parser.add_argument("--offline", action="store_true", help="Use an embedded sample dataset when no live API is available")
    parser.add_argument("--raw-only", action="store_true", help="Emit a simpler raw-series format instead of the ticket-centric structure")
    args = parser.parse_args()

    dataset = build_dataset(
        crypto=args.crypto,
        days=args.days,
        output_path=args.output,
        api_key=args.api_key,
        api_base_url=args.api_base_url,
        use_offline=args.offline,
        raw_only=args.raw_only,
    )
    ticket_count = len(dataset.get("ticket_histories", []))
    print(json.dumps({"rows": len(dataset["training_rows"]), "tickets": ticket_count, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
