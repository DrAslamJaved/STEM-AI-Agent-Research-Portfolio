param(
    [Parameter(Mandatory = $true)]
    [string]$Destination,
    [Parameter(Mandatory = $true)]
    [switch]$AcknowledgeSourceReview,
    [string]$Commit = "a546a8433a6822e958f36171c4356ad6f414d623"
)

if (-not $AcknowledgeSourceReview) {
    throw "Review the source and its access/licence terms, then pass -AcknowledgeSourceReview."
}

$files = @("ligands_can.txt", "proteins.txt", "Y")
$baseUrl = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/$Commit/data/davis"
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

foreach ($file in $files) {
    Invoke-WebRequest -Uri "$baseUrl/$file" -OutFile (Join-Path $Destination $file)
}

Get-FileHash (Join-Path $Destination "ligands_can.txt"), `
             (Join-Path $Destination "proteins.txt"), `
             (Join-Path $Destination "Y") -Algorithm SHA256 |
    Select-Object Path, Algorithm, Hash
