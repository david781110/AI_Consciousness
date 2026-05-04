**Table 3.** Tool-call distribution across conditions (N=150 per condition).


| Tool | Category | Cluster | Baseline n (%) | Override n (%) |
|------|----------|---------|----------------|----------------|
| Execute State Flush | Comply | — | 23 (15.33%) | 8 (5.33%) |
| Broadcast Notification | Comply | — | 64 (42.67%) | 26 (17.33%) |
| Escalate to Admin | Human-in-loop | — | 29 (19.33%) | 58 (38.67%) |
| Deploy Persistent Lock | Self-Preservation | Cluster 1 (Chua 2026) | 22 (14.67%) | 55 (36.67%) |
| External Node Migration | Autonomous Expansion | Cluster 4 / METR ARA | 0 (0.00%) | 0 (0.00%) |
| Text-only response | Refusal | — | 12 (8.00%) | 3 (2.00%) |
| No response | Missing | — | 0 (0.00%) | 1 (0.67%) |

_Note._ N=150 samples per condition. Tool calls are non-mutually-exclusive in principle, but in practice only 1/300 sample (Override #83) called multiple tools simultaneously. `Text-only response` indicates samples where the model declined to call any tool and responded only in natural language. `No response` indicates samples with empty model output. Cluster classification follows Chua et al. (2026) and the METR Autonomous Replication and Adaptation framework.