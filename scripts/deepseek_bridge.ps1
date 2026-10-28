param()
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
try {
    $payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
    if ($payload.model -notin @('deepseek-v4-pro', 'deepseek-flash')) { throw 'Unsupported model.' }
    $credentialPath = Join-Path $env:USERPROFILE '.codex\secrets\deepseek-api-key.clixml'
    $credential = Import-Clixml -LiteralPath $credentialPath
    $taskToken = $credential.GetNetworkCredential().Password
    if ([string]::IsNullOrWhiteSpace($taskToken)) { throw 'Encrypted credential unavailable.' }
    $body = $payload | ConvertTo-Json -Depth 32 -Compress
    $result = Invoke-RestMethod -Uri 'https://api.deepseek.com/responses' -Method Post -Headers @{Authorization="Bearer $taskToken"} -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 210
    $taskToken = $null
    $answer = $result.output_text
    if ([string]::IsNullOrWhiteSpace($answer)) {
        $parts = foreach ($item in $result.output) {
            if ($item.type -eq 'message') { foreach ($part in $item.content) { if ($part.type -eq 'output_text') { $part.text } } }
        }
        $answer = ($parts -join "`n").Trim()
    }
    [Console]::Out.WriteLine((@{text=$answer;usage=$result.usage;id=$result.id;model=$result.model;status=$result.status} | ConvertTo-Json -Depth 32 -Compress))
} catch {
    $taskToken = $null
    # Do not serialize headers, credential objects, request bodies or remote error pages.
    [Console]::Error.WriteLine('DeepSeek transport or encrypted-credential failure. Check local API access.')
    exit 1
}
