# Product commands

Run these from the repository root on Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
powershell -NoProfile -File run_demo.ps1
```

To verify the installation without starting services:

```powershell
powershell -NoProfile -File run_demo.ps1 -Verify
```

Run regression tests with:

```powershell
powershell -NoProfile -File scripts/test_local_product.ps1
```

`run_streamlit_demo_backend.py` is called by the launcher. The retained
`build_driveevidence_v3.py` and `build_driveevidence_v6.py` supply the product's
frozen geometry/indexing functions. Their names do not make them disposable
experiment scripts. Other retained compatibility helpers support runtime
imports, regression tests, frozen provenance hashes or repository hooks;
they are not daily startup steps.

The optional native/browser acceptance commands are documented in
[the product guide](../docs/FINAL_LOCAL_PRODUCT_GUIDE.md). Do not run historical
research helpers on reserved data as part of setup or daily use.

Fresh-machine setup and model downloads are documented in [INSTALL.md](../INSTALL.md).
`verify_product_install.py` is an optional native acceptance using generated media only.
