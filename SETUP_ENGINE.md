# Engine setup

Follow [INSTALL.md](INSTALL.md) for the public release, fresh-machine setup,
pinned model downloads and CPU/NVIDIA selection.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
powershell -NoProfile -File run_demo.ps1
```

Daily launch stays local. The developer entry point
`scripts/setup_local_product.ps1` also supports `-EnvironmentRoot`,
`-VerifyOnly`, `-Device cpu`, `-ModelBundle` and explicit `-Repair`.
Model, representation and selected scorer contracts remain frozen.
