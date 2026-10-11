# Multi-Provider Cloud Data Reference
## Pricing, Behavior & Negotiation Parameters for AWS, GCP and Azure
**Cloud Resource Negotiation Agent — B.Tech Project**

This reference sheet supplies the exact data points needed to build the Multi-Provider Cloud Support module
described in Phase 1 of the roadmap. It covers three areas: (1) real, current on-demand / reserved / spot pricing for
AWS, GCP, and Azure across three comparable resource tiers, (2) recommended per-provider negotiation-
behavior parameters to drive distinct agent-facing personalities in simulation, and (3) the state, reward, and
configuration schema needed to wire this data into the RL environment.
All prices are US Dollars, Linux/on-demand baseline, US primary region (us-east-1 / us-central1 / East US), gathered from
provider pricing pages and third-party pricing trackers in September 2026. Cloud prices change frequently — treat these as a
snapshot to seed the simulation, and re-verify from the official pricing pages listed at the end of each section before quoting them in
your final report.


## 1. Comparable Resource Tiers
To keep the negotiation environment fair across providers, map each provider's instance families onto three
standardized tiers by vCPU/RAM, not by name. This is the tier definition used throughout this document and
recommended for your providers.json config.

       Tier                vCPU              RAM                          Typical Use                          Representative Instances
 Small                2                4 GB                 Dev/test, light API workloads                t3.medium / e2-medium / B2s
| Medium | 2 | 8 GB | Production web apps, small DBs |
 Large                4                16 GB                ML inference, heavier services               m5.xlarge / n2-standard-4 / D4s v5


## 2. AWS Pricing Data (us-east-1, Linux)

### 2.1 On-Demand, Reserved & Spot Rates ($/hour)
         Tier / Instance                  On-Demand                    1-yr Reserved               3-yr Reserved               Spot (typical)
| Small — | $0.0416 | $0.0250 | $0.0170 | $0.0152 |
| Medium — | $0.0960 | $0.0600* | $0.0379* | $0.0290 |
| Large — | $0.1920 | $0.1200* | $0.0758* | $0.0580* |
- Estimated by applying AWS's published ~37% (1-yr) and ~60% (3-yr) Savings-Plan discount ratios to the on-demand m5 rate,
consistent with AWS's own published savings percentages for this instance family.


**Sources: aws.amazon.com/ec2/instance-types/t3 · cloudprice.net (AWS EC2) · costgoat.com/pricing/amazon-ec2 · wring.co AWS EC2 pricing guide,**
September 2026.


### 2.2 AWS Negotiation Behavior Profile
           Parameter                         Value / Setting                                             Rationale
| Concession rate | Slow (2–4% per round) | AWS is market leader; least price-sensitive |
| Patience (max rounds) | 8–10 rounds | Will let a negotiation run long before conceding |
| Price floor | ~78% of on-demand rate | Rarely beats a 1-yr Reserved-level discount |

           Parameter                          Value / Setting                                             Rationale
| Bundling | Compute + storage + support | Strong ecosystem lock-in incentive |
                                     tiers
| Volume discount | 5% at >100 units, 10% at >500 | Matches real-world enterprise agreements |


## 3. Google Cloud Platform (GCP) Pricing Data (us-central1, Linux)

### 3.1 On-Demand, Committed-Use & Spot Rates ($/hour)
         Tier / Instance                   On-Demand                      1-yr CUD                    3-yr CUD                       Spot
| Small — | $0.1120 | $0.0706 | $0.0504 | $0.0336 |
| Medium — | $0.0670 | $0.0422 | $0.0301 | $0.0617* |
| Large — | $0.2280 | $0.1224 | $0.0874 | $0.0543 |
- e2-family Spot figure shown is the published rate for the equivalent e2-standard-4 tier scaled to e2-standard-2; use it as an
approximation and confirm on the GCP pricing calculator before final submission.


**Sources: cloud.google.com/products/compute/pricing · cloudzero.com GCP Compute Engine pricing guide · usage.ai GCP Compute Engine blog ·**
cloudprice.net & economize.cloud (GCP), September 2026.


### 3.2 GCP Negotiation Behavior Profile
           Parameter                          Value / Setting                                             Rationale
| Concession rate | Moderate (5–8% per round) | Positioned as price-competitive challenger |
| Patience (max rounds) | 5–7 rounds | Moves toward a deal faster than AWS |
| Price floor | ~60% of on-demand rate | Matches its steep 1-yr/3-yr CUD discount curve |
| Bundling | Sustained-use discounts | Rewards continuous usage without upfront commitment |
                                     (automatic)
| Volume discount | Automatic Sustained-Use up to | Reflects GCP's real SUD mechanism |
                                     ~30%


## 4. Microsoft Azure Pricing Data (East US, Linux)

### 4.1 Pay-As-You-Go, Reserved & Spot Rates ($/hour)
         Tier / Instance                Pay-As-You-Go                  1-yr Reserved                3-yr Reserved                    Spot
| Small — | $0.0416 | $0.0270* | $0.0180* | $0.0125* |
| Medium — | $0.0960 | $0.0580* | $0.0379 | $0.0203 |
| Large — | $0.1920 | $0.1160* | $0.0758* | $0.0410* |
- Values marked with an asterisk are estimated by applying Azure's published Reserved Instance discount bands (up to ~72% at 3-
yr) to the confirmed pay-as-you-go rate, since per-SKU reserved/spot figures were not separately published for every tier at time of
writing.


**Sources: instances.vantage.sh (Azure VM D2s v5, B2s) · cloudprice.net (Azure Standard_D2s_v5) · amnic.com Azure VM pricing guide · spot.io Azure VM**
pricing examples, September 2026.


### 4.2 Azure Negotiation Behavior Profile
           Parameter                          Value / Setting                                             Rationale
| Concession rate | Aggressive (8–12% per round) | Most willing to discount to win enterprise deals |

          Parameter                    Value / Setting                                          Rationale
| Patience (max rounds) | 4–6 rounds | Prioritizes fast deal closure |
| Price floor | ~55% of on-demand rate | Deepest reserved-instance savings among the three (~72% max) |
| Bundling | Hybrid Benefit (bring-your- | Unique real-world differentiator for Windows/SQL workloads |
                                own-license)
| Volume discount | Negotiated Enterprise | Reflects Azure's enterprise sales motion |
                                Agreements


## 5. Cross-Provider Summary Table
This table is the single most useful reference for your RL reward design — it puts the three providers on one page
for the Medium tier (2 vCPU / 8 GB), the tier most negotiation scenarios should default to.

         Provider             On-Demand $/hr       Best Committed $/hr (3-              Spot $/hr            Concession      Price
                                                           yr)                                                 Style         Floor
| AWS (m5.large) | $0.0960 | $0.0379 | $0.0290 | Slow / stubborn | ~78% |
| GCP (e2-standard-2) | $0.0670 | $0.0301 | $0.0336 | Moderate | ~60% |
| Azure (D2s v5) | $0.0960 | $0.0379 | $0.0203 | Aggressive | ~55% |
Interpretation for your report: GCP and Azure both undercut AWS's on-demand rate for an equivalent 2 vCPU / 8 GB VM, but they
reach their floor through different mechanisms — GCP via automatic sustained-use discounts, Azure via aggressive
reserved/enterprise pricing. This asymmetry is exactly what should make your DQN agent learn provider-specific negotiation
policies rather than one generic policy.


## 6. RL State, Reward & Curriculum Data Requirements

### 6.1 Negotiation State Vector
Each negotiation episode should expose the following features to the agent's state representation:
- Current offer / counter-offer price (normalized against on-demand rate)
- Round number (0 to max_rounds)
- Provider ID (one-hot: AWS / GCP / Azure)
- Resource tier requested (Small / Medium / Large) and quantity
- Remaining patience (max_rounds − round number)
- Rolling historical success rate with this provider (optional memory feature)


### 6.2 Reward Shaping
- Primary reward: (on-demand price − final agreed price) / on-demand price
- Efficiency bonus: small positive reward for closing in fewer rounds
- Failure penalty: negative reward if provider patience is exceeded with no deal
- Per-provider floor clipping: reward cannot reflect a price below that provider's price floor (keeps the
     simulation realistic)


### 6.3 Curriculum Difficulty Bands (Phase 2)
     Difficulty              Starting Price Gap                     Provider Patience                   Simultaneous Resources
| Easy | 5–10% above floor | High (8–10 rounds) | 1 resource type |

      Difficulty         Starting Price Gap                 Provider Patience         Simultaneous Resources
| Medium | 10–20% above floor | Moderate (5–7 rounds) | 1–2 resource types |
| Hard | 20–35% above floor | Low (4–5 rounds) | 2–3 resource types |
| Expert | 35%+ above floor, random noise | Very low (3–4 rounds) | 3+ resource types, mixed providers |


## 7. Suggested `providers.json` Configuration Schema
Centralizing this data in one config file keeps the RL environment modular — a strong design point to highlight
in your report and viva. Suggested structure (values reflect Section 5 above, Medium tier):

```json
  {
      "AWS": {
        "on_demand_price": 0.0960,
        "reserved_3yr_price": 0.0379,
        "spot_price": 0.0290,
        "price_floor_ratio": 0.78,
        "concession_rate": 0.03,
        "max_rounds": 9,
        "volume_discount_tiers": [[100, 0.05], [500, 0.10]]
      },
      "GCP": {
        "on_demand_price": 0.0670,
        "reserved_3yr_price": 0.0301,
        "spot_price": 0.0336,
        "price_floor_ratio": 0.60,
        "concession_rate": 0.065,
        "max_rounds": 6,
        "sustained_use_discount_max": 0.30
      },
      "Azure": {
        "on_demand_price": 0.0960,
        "reserved_3yr_price": 0.0379,
        "spot_price": 0.0203,
        "price_floor_ratio": 0.55,
        "concession_rate": 0.10,
        "max_rounds": 5,
        "enterprise_agreement_discount_max": 0.35
      }
  }


## 8. Full Source List (September 2026)
- AWS: aws.amazon.com/ec2/instance-types/t3 — official AWS pricing page
- AWS: cloudprice.net/aws/ec2 — regional EC2 price comparison
- AWS: costgoat.com/pricing/amazon-ec2 — EC2 cost calculator, Sept 2026 snapshot
- AWS: wring.co/blog/aws-ec2-pricing-guide — On-demand/Spot comparison
- GCP: cloud.google.com/products/compute/pricing — official GCP pricing page
- GCP: cloudzero.com/blog/google-cloud-compute-engine-pricing-guide — E2/N2 rate breakdown
- GCP: usage.ai/blogs/gcp/compute-engine — CUD and Spot discount percentages
- GCP: cloudprice.net/gcp/compute & economize.cloud/resources/gcp — regional e2 pricing
- Azure: instances.vantage.sh/azure/vm — per-SKU on-demand/spot rates (D2s v5, B2s)

- Azure: cloudprice.net/vm/Standard_D2s_v5 — regional Azure VM pricing
- Azure: amnic.com/blogs/azure-vm-pricing — Reserved Instance / Savings Plan discount bands
- Azure: spot.io/resources/azure-pricing — worked PAYG/Spot/Reserved examples

```
