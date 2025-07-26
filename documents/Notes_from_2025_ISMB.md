* **Use OntoGPT for ontology-grounded entity recognition**
  *Provides a ready-made library for structured extraction with ontological IDs.*

* **Adopt a “one-entity-type-per-prompt” strategy**
  *Running separate prompts (or models) for easy vs. hard entities prevents difficult cases from degrading overall extraction accuracy.*

* **Integrate automated evaluation/unit-testing early**
  *Build a lightweight test harness so every prompt or system tweak is scored immediately.*

* **Consider converting SDRF ↔︎ LinkML**
  *Mapping SDRF to LinkML could let the pipeline emit SDRF directly (skipping interim `.ann` files) and leverage LinkML tooling.*

* **Experiment with multi-agent orchestration (Pydantic AI)**
  *Multiple cooperative agents—including a “judge” agent—may improve reliability for complex extraction tasks.*

* **Leverage PaperQA for manuscript summarization + RAG**
  *PaperQA outperforms generic LLMs on scientific texts and can feed embeddings to downstream agents.*

* **Prototype an agentic workflow**

  1. PaperQA ingests & embeds each paper.
  2. Specialized agents extract individual metadata fields.
  3. A validator/self-reflection agent reconciles outputs into a final record.

* **Expert contact for LinkML questions**
  *Carlo Kroll – [krollc@lbl.gov](mailto:krollc@lbl.gov).*

* **Explore NCBI Supplementary Material API**
  *Provides programmatic access to supplemental data associated with PubMed articles.*

