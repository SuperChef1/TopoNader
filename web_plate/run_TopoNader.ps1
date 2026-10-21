param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CommandArgs
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LocalPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$PythonExe = if ($env:TOPONADER_PYTHON) {
    $env:TOPONADER_PYTHON
}
elseif (Test-Path -LiteralPath $LocalPython) {
    $LocalPython
}
else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
        throw 'TopoNader Python interpreter not found. Create .venv or set TOPONADER_PYTHON.'
    }
    $PythonCommand.Source
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "TopoNader Python interpreter not found: $PythonExe"
}
Push-Location $ProjectRoot
try {
    & $PythonExe -m topo_nader @CommandArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
