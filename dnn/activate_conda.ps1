
$env:CONDA_EXE = "$env:USERPROFILE\Miniconda3\Scripts\conda.exe"
$env:_CE_M = ""
$env:_CE_CONDA = ""
$env:_CONDA_ROOT = "$env:USERPROFILE\Miniconda3"
$env:_CONDA_EXE = "$env:USERPROFILE\Miniconda3\Scripts\conda.exe"

$env:Path = "$env:USERPROFILE\Miniconda3;$env:USERPROFILE\Miniconda3\Scripts;$env:USERPROFILE\Miniconda3\Library\bin;$env:Path"

function conda {
    $condaPath = "$env:USERPROFILE\Miniconda3\Scripts\conda.exe"
    if ($args[0] -eq "activate") {
        & cmd /c "`"$condaPath`" activate $($args[1..$args.Length]) && set" |
            ForEach-Object {
                if ($_ -match "^(.*?)=(.*)$") {
                    Set-Item -Path "env:$($matches[1])" -Value $matches[2]
                }
            }
    } else {
        & cmd /c "`"$condaPath`" $args"
    }
}

Write-Host "Conda has been activated for this session."
Write-Host "You can now use 'conda' commands."
