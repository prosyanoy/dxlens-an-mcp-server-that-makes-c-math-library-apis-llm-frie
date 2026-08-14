Model: `scripted (deterministic, context-grounded)`; 30 tasks; pass = compiles + runs + exact stdout.

| Context | Budget (tokens) | Context size | Compiles | Pass@1 | Pass with DxLens compile loop (<=2 retries) |
|---|---|---|---|---|---|
| raw header | unlimited | 4595 | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| raw header | 3000 | 2994 | 13% (4/30) | 13% (4/30) | 90% (27/30) |
| raw header | 1500 | 1499 | 3% (1/30) | 3% (1/30) | 67% (20/30) |
| raw header | 750 | 745 | 3% (1/30) | 3% (1/30) | 67% (20/30) |
| DxLens spec (llms.txt) | unlimited | 1698 | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| DxLens spec (llms.txt) | 3000 | 1698 | 100% (30/30) | 100% (30/30) | 100% (30/30) |
| DxLens spec (llms.txt) | 1500 | 1491 | 90% (27/30) | 90% (27/30) | 90% (27/30) |
| DxLens spec (llms.txt) | 750 | 744 | 3% (1/30) | 3% (1/30) | 80% (24/30) |
