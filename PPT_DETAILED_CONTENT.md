# Detailed PPT Content Script

## How to use this document
This document is written as a presentation script you can directly convert into slides. For each slide, you will find:
- Suggested slide title
- Slide bullets (what to show on screen)
- Detailed speaking content (what to say)
- Optional visual suggestions

You can shorten each speaking section while building your final PPT. This script intentionally contains rich detail so you can summarize according to your audience.

---

## Slide 1 - Title and Opening Context

### Slide title
Advanced AML Detection with Temporal Graph Learning and Ensemble Intelligence

### Slide bullets
- Financial crime analytics project
- Graph + temporal + classical ML fusion
- Real-time fraud risk scoring API
- End-to-end system: data -> model -> deployment

### Speaking content
Today I will present an advanced anti-money laundering detection system that combines traditional machine learning, graph neural networks, and temporal deep learning into one unified pipeline. The motivation is simple: money laundering behavior is not only about suspicious amounts; it is about hidden relationships, transaction sequences, and timing patterns across many accounts. A single transaction may look normal in isolation, but when seen as part of a dynamic network, it can become highly suspicious.

This project was built as a full stack machine learning system. It begins with transaction-level data, constructs account interaction graphs, engineers domain-specific AML features, trains multiple model families, and deploys an inference service using Flask with an interactive web interface. It supports both experimentation and practical usage.

The core contribution is not only one model but a robust architecture where each model contributes different strengths: baseline models capture direct feature signals, GraphSAGE captures neighborhood context, and Temporal Graph Network captures sequence and time dynamics. The final decision is made through a learned weighted ensemble, creating stronger and more stable fraud detection performance.

### Visual suggestion
Use a clean architecture banner: Data Layer -> Feature Layer -> Model Layer -> API Layer.

---

## Slide 2 - Problem Statement: Why AML Detection is Hard

### Slide title
The Detection Challenge in Modern Financial Networks

### Slide bullets
- Rule-based systems miss evolving patterns
- False positives overload investigators
- Fraudsters exploit structure and timing
- Need precision, recall, and explainability

### Speaking content
Anti-money laundering is one of the hardest applied machine learning problems because malicious actors continuously adapt. Classical rule systems, for example fixed threshold alerts, can catch obvious events but fail to detect creative laundering pathways such as layering through multiple intermediaries, structuring transactions just below reporting limits, and burst transfers during low-monitoring windows.

A major operational issue is false alert volume. Even a highly accurate system can generate thousands of non-actionable alerts in high-volume environments, overwhelming compliance teams. This reduces trust in the system and delays investigation of truly high-risk entities.

Another challenge is context fragmentation. Traditional tabular models treat each transaction as independent or only lightly connected through manually aggregated account features. In reality, laundering is network behavior. The sender and receiver are embedded in a larger graph of counterparties, circular flows, and temporal cascades.

Therefore, an effective AML platform must combine three capabilities: structural awareness of account relationships, temporal awareness of transaction sequences, and calibrated probabilistic outputs that can map cleanly to risk tiers. This project addresses exactly those three requirements in one production-oriented design.

### Visual suggestion
Add a 2x2 matrix: easy-to-detect vs hard-to-detect patterns, and static vs temporal patterns.

---

## Slide 3 - Project Objectives and Success Criteria

### Slide title
What This Project Set Out to Achieve

### Slide bullets
- Build multi-model AML detector
- Engineer domain-specific high-value features
- Compare baseline, graph, and temporal models
- Deploy real-time API with risk classes

### Speaking content
The project had four objective layers. First, model diversity: instead of relying on a single architecture, we train baseline models, graph neural networks, and a temporal graph network. This allows direct performance comparison and robustness through ensembling.

Second, feature sophistication: generic financial features are not enough for AML. We engineered specialized indicators such as near-threshold ratio for structuring behavior, round-number ratio for suspicious payment shaping, cycle participation for layered fund movement, velocity and burst metrics for abnormal activity spikes, and rolling window behavior across 1-hour, 6-hour, and 24-hour horizons.

Third, practical evaluation: performance is assessed with accuracy, precision, recall, F1, and ROC-AUC, ensuring we balance both quality of alerts and missed fraud risk. In AML, recall matters because missed fraud is costly, while precision matters because analyst capacity is finite.

Fourth, deployment realism: the system includes a Flask API with endpoints for prediction, health checks, and model information. Output is not just a score but a risk classification level and per-model contribution values, enabling operational adoption and interpretability.

### Visual suggestion
Create a four-pillar diagram: Modeling, Features, Evaluation, Deployment.

---

## Slide 4 - Data Foundation and Scope

### Slide title
Dataset Snapshot and Experimental Scope

### Slide bullets
- Source file: synthetic AML transaction dataset
- Total records: 6000
- Unique accounts: 500
- Label ratio: 12.0% fraud

### Speaking content
The training and evaluation foundation is a synthetic AML transaction dataset designed to mimic realistic transfer patterns and suspicious behavior signatures. The dataset contains 6000 labeled transactions with a 12.0% fraud ratio, meaning 720 fraudulent events and 5280 legitimate events. While this class balance is still imbalanced, it is not extremely sparse, which allows stable experimentation across multiple model families.

Entity coverage is broad: the network includes 500 unique accounts when combining sender and receiver identities. This creates enough graph connectivity to evaluate neighborhood-based reasoning while keeping the computational scale practical for rapid iteration.

The project uses a deterministic split into 4800 training transactions and 1200 test transactions, preserving the same 12.0% fraud proportion in both sets. This ensures fair comparison among all model variants.

From a project communication perspective, this dataset size is ideal for demonstrations: it is large enough to show meaningful graph patterns and temporal behavior, but small enough to train quickly and iterate architecture choices. The final deck should emphasize that this system architecture is scalable, even though current experiments were intentionally run on a compact benchmark for rapid end-to-end validation.

### Visual suggestion
Show a compact table with rows, accounts, fraud count, train/test split.

---

## Slide 5 - Transaction Behavior Profile

### Slide title
Transaction Composition and Behavioral Distribution

### Slide bullets
- transfer: 41.88%
- payment: 26.97%
- cash_out: 12.47%
- deposit: 9.85%, withdrawal: 8.83%

### Speaking content
Transaction type distribution gives important context to model behavior. In this dataset, transfer transactions dominate at 41.88%, followed by payment at 26.97%. Cash out transactions account for 12.47%, deposits 9.85%, and withdrawals 8.83%.

This profile matters because laundering activity often blends into common transaction categories rather than appearing only in rare types. A robust detector cannot overfit to simple type-based priors. Instead, it must look for combinations of amount dynamics, graph routes, timing, and counterpart diversity within each type.

Amount statistics also reveal scale behavior: minimum amount is 5.00, mean is 162.70, median is 101.47, and maximum is 4552.88. The difference between mean and median indicates moderate right skew, which is typical in transaction systems where occasional larger events shift averages.

For presentation, emphasize that suspicious behavior does not require extreme amounts. Many laundering tactics intentionally avoid obvious thresholds. This justifies why this project heavily invests in structuring and temporal progression features rather than only absolute amount triggers.

### Visual suggestion
Use two charts: a bar chart for transaction types and a box/violin chart for transaction amounts.

---

## Slide 6 - Feature Engineering Philosophy

### Slide title
From Raw Transactions to AML Intelligence Features

### Slide bullets
- 67 transaction-level engineered features
- 31 node-level account features
- 9 edge features for temporal graph models
- Domain + graph + time feature fusion

### Speaking content
A major strength of this system is deep feature engineering grounded in AML reasoning. The processed metadata shows 67 transaction-level features used for prediction, 31 node-level account descriptors, and 9 edge-level temporal interaction features for graph models.

Feature design was not random expansion. It follows a layered strategy:
1) Transaction semantics: amount scaling, transaction type encoding, cyclic time encodings for hour and day of week, and weekend flags.
2) Account behavior: in/out degree, in/out volume, average in/out amount, and counterparty diversity.
3) Graph structure: PageRank, cycle participation, and flow imbalance.
4) AML-specific behavioral signatures: near-threshold ratio, round-number ratio, variance regularity, and combined structuring score.
5) Temporal intensity windows: rolling maxima for count, volume, and counterparties over 1h, 6h, and 24h.

This layered approach improves separability between legitimate but active accounts and truly suspicious behavioral profiles. In practical terms, better features reduce model complexity required for a strong signal, improve interpretability, and support resilient performance when fraudsters shift one pattern but cannot hide all dimensions simultaneously.

### Visual suggestion
Feature map pyramid with five layers from raw to AML-specialized.

---

## Slide 7 - Graph Construction Strategy

### Slide title
Modeling Financial Flows as a Temporal Directed Graph

### Slide bullets
- Nodes: accounts
- Edges: transactions with attributes
- Directed graph preserves money flow direction
- Temporal ordering retained for sequence models

### Speaking content
Graph construction converts transactional logs into a relational learning problem. Each account is represented as a node. Each transaction becomes a directed edge from sender to receiver with attached attributes such as scaled amount, type encoding, and time-based cyclic features.

Directionality is critical in AML. Outflow and inflow behavior carry different risk meaning. For example, an account with diffuse outflow to many receivers may indicate layering, while concentrated inflow from many weakly connected senders can indicate funneling.

Temporal ordering is equally important. For the temporal models, edge events are processed in chronological sequence, allowing the model to observe progression effects such as rapid relay chains, sudden reactivation after dormancy, or short-window bursts around specific hours.

Graph perspective provides capabilities that tabular-only systems cannot replicate efficiently: neighborhood aggregation, path-level context, and structural anomaly sensitivity. Even when a single edge appears benign, graph position can amplify suspicion if it participates in cyclic routes or interfaces with high-risk subnetworks.

In the deck, frame this as a shift from transaction classification to behavior ecosystem modeling, which is closer to how financial investigators reason about laundering patterns in practice.

### Visual suggestion
Show a mini directed network with 8-10 nodes and highlighted suspicious cycle.

---

## Slide 8 - Baseline Models and Their Role

### Slide title
Baseline Layer: Logistic Regression and Random Forest

### Slide bullets
- Logistic Regression for linear baseline
- Random Forest for nonlinear tabular power
- Fast training and strong interpretability anchors
- Baselines provide calibration reference

### Speaking content
The baseline layer consists of Logistic Regression and Random Forest. This is important because strong baselines create honest comparisons. If advanced models cannot beat high-quality baseline systems, architectural complexity is unjustified.

Logistic Regression achieved high recall and stable ROC performance, reflecting the value of engineered features even in a linear framework. Random Forest delivered stronger precision and F1 by capturing nonlinear interactions between amount behavior, structuring indicators, and temporal descriptors.

In production strategy, baseline models are not throwaway components. They often serve as robust fallback paths, low-latency checkers, and explainability references. Tree-based models also support feature importance analysis that can guide future feature refinement.

For this project, Random Forest becomes the baseline representative used inside the final ensemble. This choice is data-driven because it outperformed Logistic Regression on F1 and precision while maintaining excellent ROC-AUC.

When presenting this section, emphasize discipline: advanced deep models were introduced after establishing tabular strength, not instead of it. This strengthens the credibility of the final architecture and shows engineering maturity in model development workflow.

### Visual suggestion
Simple model card table with strengths, weaknesses, and use in final ensemble.

---

## Slide 9 - GraphSAGE Model Design

### Slide title
Temporal GraphSAGE: Neighborhood-Aware Fraud Learning

### Slide bullets
- GraphSAGE captures local network context
- Uses node and edge attributes
- Hidden dimension around 64 in deployment model
- Strong balance of performance and simplicity

### Speaking content
GraphSAGE extends beyond tabular assumptions by learning from node neighborhoods. Instead of treating sender and receiver as isolated feature vectors, it aggregates relational context from nearby nodes and edges. This allows representation learning that captures structural patterns such as repeated interaction motifs and local fraud clusters.

In this implementation, GraphSAGE integrates node descriptors with temporal edge attributes and feeds combined representations into an edge-level prediction head for fraud probability scoring. The deployed configuration uses a compact hidden dimension, enabling practical inference with meaningful graph intelligence.

GraphSAGE performed very strongly in experiments, delivering high precision and recall balance. It became the largest contributor in the learned ensemble weights, which indicates its representations carry significant predictive value on this dataset.

Operationally, GraphSAGE often provides a sweet spot between tabular models and more complex temporal memory architectures. It is expressive, reasonably interpretable through neighborhood-level analysis, and computationally manageable.

For slide narration, communicate that GraphSAGE is the relational backbone: it lifts detection from individual events to local topology awareness, which is essential for uncovering laundering routes that evade amount-only or rule-only surveillance.

### Visual suggestion
Diagram: sender and receiver nodes with neighborhood aggregation arrows feeding classifier.

---

## Slide 10 - Temporal Graph Network (TGN) Architecture

### Slide title
TGN: Time2Vec + Memory + Temporal Attention

### Slide bullets
- Memory dimension: 64
- Time encoding dimension: 16
- Hidden dimension: 128
- Multi-head attention: 4 heads, 2 layers

### Speaking content
The Temporal Graph Network is the most advanced model in this system. It is designed to model not only structure but also the evolution of interactions over time. The architecture combines multiple components:

1) Time2Vec encoding transforms timestamps into learnable temporal representations, capturing both linear and periodic effects.
2) A memory mechanism tracks state evolution for nodes, allowing historical context to influence current predictions.
3) Multi-head temporal attention focuses on relevant interaction history with time-aware weighting.
4) Stacked layers refine representations for downstream fraud prediction.

The deployed configuration uses memory dimension 64, time dimension 16, hidden dimension 128, four attention heads, and two layers. These settings provide sufficient capacity without excessive over-parameterization for this dataset scale.

Although TGN is architecturally powerful, its standalone performance in this benchmark is lower than GraphSAGE and Random Forest. This is not uncommon on moderate-size synthetic datasets where very deep temporal modeling may require more event diversity or longer history depth to dominate. Still, TGN provides complementary signal diversity, which helps the ensemble produce stronger final recall and F1.

### Visual suggestion
Layered architecture block: event stream -> Time2Vec -> memory update -> attention -> classifier.

---

## Slide 11 - Ensemble Learning Strategy

### Slide title
Why the Ensemble Wins

### Slide bullets
- Learned weighted fusion of model probabilities
- Weights learned from validation behavior
- Optional isotonic calibration for reliability
- Final output tuned for operational risk scoring

### Speaking content
The best-performing detector is the learned ensemble. Instead of averaging model outputs equally, the pipeline learns optimal weights for baseline, GraphSAGE, and TGN probabilities. Current learned weights are approximately:
- baseline: 0.424
- GraphSAGE: 0.483
- TGN: 0.093

These weights reveal a useful insight: GraphSAGE contributes the strongest primary signal, baseline contributes substantial complementary information, and TGN contributes a smaller but still useful correction signal.

After weighted combination, optional isotonic calibration can be applied to improve probability reliability. Calibration is crucial in compliance settings because thresholds trigger workflows, escalations, and resource allocation. A score of 0.80 should correspond to genuinely higher observed risk than 0.60 in a statistically meaningful way.

This ensemble strategy improves robustness. If one model underperforms under certain pattern regimes, others can stabilize prediction quality. In deployment-oriented terms, ensemble design reduces dependence on a single inductive bias and better handles heterogeneous fraud behavior.

For the audience, the key message is that architecture diversity plus learned fusion yields stronger outcomes than single-model dependence.

### Visual suggestion
Weighted sum equation and a stacked contribution bar chart.

---

## Slide 12 - Quantitative Performance Comparison

### Slide title
Model Results: Precision, Recall, F1, ROC-AUC

### Slide bullets
- Logistic Regression F1: 0.8225
- Random Forest F1: 0.8655
- GraphSAGE F1: 0.8897
- TGN F1: 0.8148
- Ensemble F1: 0.9116 (best)

### Speaking content
Performance comparison confirms the value of model progression and fusion. Logistic Regression sets a respectable baseline with F1 of 0.8225 and very high recall. Random Forest improves to F1 0.8655 and high precision, showing strong nonlinear tabular learning.

GraphSAGE further improves to F1 0.8897 with excellent ROC-AUC near 0.995, validating the benefit of graph neighborhood information. TGN achieves F1 0.8148 and ROC-AUC around 0.975. While lower in this specific benchmark, it contributes temporal diversity useful for ensemble behavior.

The ensemble achieves the best headline result: accuracy 0.9783, precision 0.8933, recall 0.9306, F1 0.9116, and ROC-AUC 0.9945. This is the model package recommended for deployment.

A strong presentation move is to show not just best metric but metric balance. In AML settings, high recall helps avoid missed fraud while high precision keeps analyst workload manageable. The ensemble offers a favorable compromise and demonstrates clear practical value.

### Visual suggestion
Use grouped bar chart for precision, recall, F1 per model and a separate ROC-AUC line or bars.

---

## Slide 13 - Operational Interpretation of Metrics

### Slide title
What These Numbers Mean in Investigation Terms

### Slide bullets
- Test size: 1200 transactions
- Fraud positives in test: 144
- Ensemble recall: 93.06%
- Ensemble precision: 89.33%

### Speaking content
To make metrics operationally meaningful, translate percentages into approximate case counts. On the 1200-sample test set with 12% fraud prevalence, there are 144 fraudulent transactions.

With recall 93.06%, the ensemble identifies about 134 of those fraud events and misses about 10. With precision 89.33%, the total predicted suspicious set is about 150 transactions, of which roughly 16 are false positives.

This is a favorable operating profile: high fraud capture with manageable alert noise. Compare this to many rule-heavy systems where false positives dominate and analyst throughput becomes a bottleneck.

In real AML workflows, these numbers influence staffing and review cost. If alert volume is too high, cases are delayed or triaged aggressively. If recall is low, major laundering events can pass undetected. Balanced performance, as seen here, supports practical deployment discussions.

When presenting, emphasize that model performance should always be evaluated in terms of investigation economics and compliance risk, not only abstract leaderboard metrics. This makes your PPT stronger for both academic and industry audiences.

### Visual suggestion
Small confusion matrix with approximate counts: TP 134, FP 16, FN 10, TN 1040.

---

## Slide 14 - Risk Scoring and Decision Policy

### Slide title
From Probability to Actionable Risk Tiers

### Slide bullets
- critical: p >= 0.85
- high: 0.70-0.84
- medium: 0.45-0.69
- low: 0.25-0.44
- minimal: p < 0.25

### Speaking content
Model outputs become useful only when mapped to operational action. This system includes a five-level risk policy based on final ensemble probability. The thresholds are:
- Critical for scores at or above 0.85
- High for 0.70 to 0.84
- Medium for 0.45 to 0.69
- Low for 0.25 to 0.44
- Minimal below 0.25

This granular policy enables differentiated response. Critical cases can trigger immediate escalation and enhanced due diligence. High and medium can route to prioritized analyst queues. Low cases may be monitored with lighter workflows. Minimal cases remain logged for trend monitoring and future model learning.

A key deployment principle is threshold governance. As compliance priorities or workload constraints change, thresholds can be tuned without retraining core models. The same calibrated probability backbone supports dynamic policy adjustment.

In the PPT, explain that this separation of model inference and decision policy is a design advantage. It keeps the machine learning core stable while allowing risk operations teams to adapt controls through policy configuration.

### Visual suggestion
Risk ladder graphic with color-coded tiers and suggested action labels.

---

## Slide 15 - API and Deployment Architecture

### Slide title
Production-Ready Inference Interface

### Slide bullets
- Flask API with CORS support
- Endpoints: /health, /predict, /model-info
- Loads preprocessors, embeddings, and trained models
- Supports web UI and service integration

### Speaking content
The deployment layer is implemented as a Flask service designed for practical integration. It exposes key endpoints:
- /health for service heartbeat checks
- /predict for transaction risk inference
- /model-info for model metadata, weights, and risk threshold visibility

At startup, the service loads preprocessing assets, node embedding artifacts, baseline model weights, GraphSAGE checkpoint, optional TGN checkpoint, ensemble weights, and optional calibrator object. This design ensures prediction consistency with training-time transformations.

During inference, the API validates payload format, normalizes transaction type, checks timestamp compatibility, builds feature vectors, runs all available models, computes ensemble output, optionally calibrates probability, and returns both per-model and final risk outputs.

This is valuable because many academic projects stop at notebook metrics. Here, the system reaches deployable form with repeatable input/output behavior and clear contract boundaries, making it suitable for demonstrations, integration pilots, and future cloud deployment.

### Visual suggestion
Request-response flow diagram with input JSON and output JSON fields.

---

## Slide 16 - Explainability and Trust Considerations

### Slide title
Building Analyst Trust in Model Decisions

### Slide bullets
- Multi-model output transparency
- Feature-rich context behind scores
- Risk tier mapping simplifies triage
- Model-info endpoint supports governance

### Speaking content
Explainability in AML is not optional. Investigators and compliance auditors need to understand why a transaction was flagged. This project supports trust in several ways.

First, it returns multiple model probabilities, not only one opaque score. Analysts can see whether risk is supported by tabular features, graph context, or temporal behavior.

Second, engineered features are behaviorally meaningful: near-threshold ratio, burst score, cycle participation, and flow imbalance can be interpreted in relation to known laundering typologies.

Third, risk tiers convert probabilities into operational semantics. Teams can align review effort with risk class and maintain transparent policies.

Fourth, model metadata endpoints provide visibility into active models and ensemble weighting, supporting model governance and documentation requirements.

For your presentation, position this as "practical explainability." It does not claim complete causal interpretability, but it offers enough structured transparency for operational decision support and compliance reporting. This balance is realistic and typically preferred in production fraud systems where speed, accuracy, and accountability must coexist.

### Visual suggestion
Callout dashboard mockup: model probabilities + risk class + top contributing feature categories.

---

## Slide 17 - Technical Deep Dive: Core Equations and Learning View

### Slide title
Mathematical View of Detection Pipeline

### Slide bullets
- Weighted ensemble probability
- Precision, recall, F1 definitions
- Time encoding as learnable transformation
- Graph message aggregation intuition

### Speaking content
A compact mathematical narrative strengthens the technical credibility of the deck.

Final ensemble probability is computed as a weighted sum of model outputs:
P_final = w_b * P_baseline + w_g * P_graphsage + w_t * P_tgn
where current weights are approximately w_b = 0.424, w_g = 0.483, w_t = 0.093.

Evaluation metrics are defined as:
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)

These equations matter because AML performance must optimize both alert quality and fraud capture.

At representation level, GraphSAGE learns node embeddings by aggregating neighborhood information, while TGN augments this process with temporal encoding and memory updates, enabling sequence-sensitive behavior modeling.

If you include this slide, keep formulas simple and connect each equation to practical impact. The goal is not to overload with theory but to show rigorous grounding behind engineering decisions.

### Visual suggestion
Left side formulas, right side interpretation boxes ("investigator impact").

---

## Slide 18 - Strengths, Trade-offs, and Critical Reflection

### Slide title
What Worked Well and What Needs Improvement

### Slide bullets
- Ensemble outperforms all single models
- Feature engineering strongly boosted baseline power
- Graph models captured relational behavior
- TGN underperformed as standalone on this benchmark

### Speaking content
A strong presentation includes honest critical reflection. The major success is clear: the ensemble delivers best overall balance and strongest F1 score. This validates the multi-model architecture strategy.

Another success is feature engineering quality. Even baseline models performed well, indicating high signal quality in engineered features. This is important because good features increase system resilience and interpretability.

GraphSAGE performance confirms the value of relational context in AML detection. It became the dominant contributor in ensemble weighting.

A trade-off appears with standalone TGN performance on this dataset. Reasons may include dataset scale, synthetic pattern complexity limits, sequence depth constraints, or hyperparameter alignment. However, lower standalone score does not make TGN useless; it still contributes diversity to ensemble output and acts as a future-ready module for larger temporal workloads.

In your talk, frame this as mature ML practice: not every advanced model wins alone on every dataset, but architecture-level design can still convert complementary behavior into a better system outcome.

### Visual suggestion
Two-column table: Strengths vs Limitations with mitigation notes.

---

## Slide 19 - Real-World Implementation Roadmap

### Slide title
From Project Prototype to Enterprise AML Platform

### Slide bullets
- Integrate with real transaction streams
- Add concept drift and retraining pipelines
- Expand explainability and case management hooks
- Strengthen governance and monitoring

### Speaking content
To transition this project toward enterprise readiness, the roadmap can follow phased execution.

Phase 1: Data and integration. Replace synthetic source with secure real transaction feeds, customer/account context, and external watchlist signals. Add robust schema validation and feature store versioning.

Phase 2: MLOps and monitoring. Implement periodic retraining, drift detection, calibration monitoring, and threshold governance dashboards. Fraud behavior changes over time, so monitoring is as important as initial training.

Phase 3: Explainability and investigator workflow. Add reason codes, feature attribution summaries, and direct integration with case management systems. This helps convert model outputs into actionable investigations with audit trails.

Phase 4: Risk and compliance hardening. Document model cards, approval gates, rollback procedures, and testing standards for regulatory review.

This roadmap shows your audience that the project is not only technically strong but also aligned with production lifecycle realities in regulated financial environments.

### Visual suggestion
Timeline with four phases and expected outcomes.

---

## Slide 20 - Business Impact and Value Proposition

### Slide title
Why This System Matters Beyond Technical Performance

### Slide bullets
- Better fraud capture with manageable alert noise
- Faster prioritization through risk tiers
- Lower investigative waste
- Strong foundation for scalable compliance analytics

### Speaking content
The business value of this system can be summarized in three outcomes.

First, improved detection effectiveness: high recall captures more suspicious activity before funds disappear across layered routes.

Second, improved operational efficiency: high precision and tiered risk classification help compliance teams prioritize high-impact cases and reduce time spent on low-value alerts.

Third, improved institutional resilience: combining graph and temporal intelligence supports adaptation against evolving laundering strategies that bypass static rule systems.

Even in this controlled benchmark, the architecture demonstrates clear practical promise. Ensemble F1 above 0.91 with strong ROC-AUC indicates robust ranking quality and balanced error behavior.

For stakeholder audiences, connect these results to concrete effects: fewer missed high-risk events, more analyst throughput, better audit confidence, and a reusable machine learning platform for broader financial crime use cases.

### Visual suggestion
Impact map: Detection Quality -> Analyst Efficiency -> Compliance Confidence.

---

## Slide 21 - Suggested Demo Flow During Presentation

### Slide title
Live Demo Storyboard (Optional)

### Slide bullets
- Show API health and model metadata
- Submit normal transaction
- Submit suspicious transaction
- Compare per-model probabilities and final risk class

### Speaking content
A short demo can dramatically improve audience engagement. Start by showing service readiness through the health endpoint. Then show model metadata so the audience sees active models, weights, and risk thresholds.

Next, submit a likely normal transaction and observe low or minimal risk output. Then submit a crafted suspicious pattern transaction, for example one with transfer type, unusual timing, and high structuring-like behavior context if available. Highlight how each model probability shifts and how the ensemble converts those signals into a higher risk class.

The key teaching moment is model diversity: baseline might react to amount and type features, GraphSAGE may react to network context, and TGN may react to temporal progression. The ensemble then synthesizes these views into one decision.

Keep demo duration under three minutes and focus on interpretive value rather than UI effects. The purpose is to prove that the full pipeline is not theoretical but executable and decision-ready.

### Visual suggestion
One screenshot per step with callouts for key fields.

---

## Slide 22 - Conclusion and Closing Message

### Slide title
Final Takeaways

### Slide bullets
- End-to-end AML intelligence system delivered
- Strong ensemble performance: F1 0.9116
- Graph and temporal learning add practical value
- Clear path from prototype to real deployment

### Speaking content
To conclude, this project demonstrates an end-to-end anti-money laundering detection system that is both technically rigorous and deployment-aware. It combines strong feature engineering, graph intelligence, temporal modeling, and ensemble optimization into a coherent architecture.

The quantitative result is compelling: ensemble F1 of 0.9116 with high recall and strong precision, supported by near-excellent ROC-AUC. More importantly, the system outputs actionable risk classes and transparent multi-model scores suitable for compliance workflows.

The project also shows disciplined engineering practice: baseline benchmarking, modular training pipeline, artifact management, API deployment, and clear model reporting. This is not only a modeling exercise but a practical analytics system design.

Finally, the roadmap toward enterprise extension is clear: real data integration, MLOps monitoring, explainability deepening, and governance hardening. That means the work has immediate educational value and realistic production potential.

If you summarize this deck for your final PPT, focus on one core message: effective AML detection requires combining feature intelligence, graph relationships, and temporal behavior into a single robust decision framework.

### Visual suggestion
Closing slide with architecture recap and three key numbers: 6000 transactions, 500 accounts, F1 0.9116.

---

## Appendix A - Performance Table (ready to paste into PPT)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9500 | 0.7165 | 0.9653 | 0.8225 | 0.9896 |
| Random Forest | 0.9692 | 0.9084 | 0.8264 | 0.8655 | 0.9941 |
| Temporal GraphSAGE | 0.9742 | 0.9124 | 0.8681 | 0.8897 | 0.9950 |
| TGN | 0.9542 | 0.7908 | 0.8403 | 0.8148 | 0.9754 |
| Ensemble | 0.9783 | 0.8933 | 0.9306 | 0.9116 | 0.9945 |

---

## Appendix B - Data Table (ready to paste into PPT)

| Data Attribute | Value |
|---|---:|
| Total transactions | 6000 |
| Fraud transactions | 720 |
| Fraud ratio | 12.00% |
| Unique sender IDs | 500 |
| Unique receiver IDs | 500 |
| Unique combined accounts | 500 |
| Train size | 4800 |
| Test size | 1200 |
| Train fraud ratio | 12.00% |
| Test fraud ratio | 12.00% |

Transaction type distribution:
- transfer: 2513 (41.88%)
- payment: 1618 (26.97%)
- cash_out: 748 (12.47%)
- deposit: 591 (9.85%)
- withdrawal: 530 (8.83%)

Amount statistics:
- min: 5.00
- mean: 162.70
- median: 101.47
- max: 4552.88

---

## Appendix C - Feature and System Facts

- Transaction-level engineered features: 67
- Node-level account features: 31
- Edge features for graph models: 9
- Ensemble weights:
  - baseline: 0.423963
  - graphsage: 0.483431
  - tgn: 0.092606
- API endpoints:
  - GET /health
  - POST /predict
  - GET /model-info
- Risk policy thresholds:
  - critical: p >= 0.85
  - high: 0.70 <= p < 0.85
  - medium: 0.45 <= p < 0.70
  - low: 0.25 <= p < 0.45
  - minimal: p < 0.25
