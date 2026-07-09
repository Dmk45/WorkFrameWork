import requests

url = "https://external-api.kalshi.com/trade-api/v2/markets"

cursor = None
found = 0

while True:

    params = {
        "limit": 100
    }

    if cursor:
        params["cursor"] = cursor

    r = requests.get(url, params=params)

    print("Request status:", r.status_code)

    r.raise_for_status()

    data = r.json()

    for market in data["markets"]:

        ticker = market.get("ticker", "")
        title = market.get("title", "")
        subtitle = market.get("subtitle", "")

        search = (
            ticker +
            " " +
            title +
            " " +
            subtitle
        ).upper()

        if (
            "BTC" in search
            or "BITCOIN" in search
            or "CRYPTO" in search
        ):

            found += 1

            print("--------------------------------")
            print("Ticker:", ticker)
            print("Title:", title)
            print("YES:", market.get("yes_ask_dollars"))
            print("NO:", market.get("no_ask_dollars"))
            print("Expiration:", market.get("expiration_time"))


    cursor = data.get("cursor")

    if not cursor:
        break

print("\nBTC markets found:", found)