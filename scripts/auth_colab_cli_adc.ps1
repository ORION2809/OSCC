$ErrorActionPreference = "Stop"

Write-Host "This opens a one-time Google login for Colab CLI ADC credentials."
Write-Host "Use the same Google account you use for Colab."

wsl -d Ubuntu -- bash -lc "~/google-cloud-sdk/bin/gcloud auth application-default login --no-launch-browser --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory"

if ($LASTEXITCODE -ne 0) {
    throw "gcloud ADC login failed with exit code $LASTEXITCODE"
}

Write-Host "Verifying Colab CLI auth..."
wsl -d Ubuntu -- bash -lc "source ~/.local/bin/env && colab --auth=adc sessions"

if ($LASTEXITCODE -ne 0) {
    throw "Colab CLI auth verification failed with exit code $LASTEXITCODE"
}

Write-Host "Colab CLI ADC auth is ready."
