$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$target = (Get-ChildItem -LiteralPath $root -Filter '*_*.docx' | Select-Object -First 1).FullName
$qa = Join-Path $root 'report_qa'
New-Item -ItemType Directory -Force -Path $qa | Out-Null
$pdf = Join-Path $qa 'report.pdf'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $document = $word.Documents.Open($target, $false, $true)
    $document.Repaginate()
    Write-Output ('Pages: ' + $document.ComputeStatistics(2))
    $word.ActiveWindow.View.Type = 3
    Add-Type -AssemblyName System.Drawing
    $pages = $word.ActiveWindow.ActivePane.Pages
    for ($i = 1; $i -le $pages.Count; $i++) {
        $page = $pages.Item($i)
        $emfPath = Join-Path $qa ('page-' + $i + '.emf')
        [System.IO.File]::WriteAllBytes($emfPath, [byte[]]$page.EnhMetaFileBits)
        $emf = [System.Drawing.Image]::FromFile($emfPath)
        $bmp = New-Object System.Drawing.Bitmap(1191, 1684)
        $graphics = [System.Drawing.Graphics]::FromImage($bmp)
        $graphics.Clear([System.Drawing.Color]::White)
        $graphics.DrawImage($emf, 0, 0, 1191, 1684)
        $bmp.Save((Join-Path $qa ('page-' + $i + '.png')), [System.Drawing.Imaging.ImageFormat]::Png)
        $graphics.Dispose()
        $bmp.Dispose()
        $emf.Dispose()
        Write-Output ('Rendered page ' + $i)
    }
}
finally {
    if ($null -ne $document) { $document.Close(0) }
    if ($null -ne $word) { $word.Quit() }
}
