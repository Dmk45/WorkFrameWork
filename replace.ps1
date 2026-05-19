$files = Get-ChildItem -Path "src", "tests" -Filter "*.zig" -Recurse
foreach ($file in $files) {
    if ($file.FullName -match "autodiff.zig") { continue }
    $content = [System.IO.File]::ReadAllText($file.FullName)
    $content = $content.Replace("std.ArrayList", "std.array_list.Managed")
    [System.IO.File]::WriteAllText($file.FullName, $content)
}
