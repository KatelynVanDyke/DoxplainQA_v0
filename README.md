# DoxplainQA: A Unified Question–Answering Dataset

## Overview

**DoxplainQA** is a **unified question–answering (QA) dataset** constructed to support **systematic evaluation, comparison, and explanation** of QA models across heterogeneous source datasets. The dataset harmonizes multiple established QA benchmarks into a **single, normalized schema**, enabling **consistent training, inference, and evaluation** pipelines within the Doxplain framework.

The primary design goals are:

- Schema uniformity across diverse QA datasets  
- Minimal but sufficient fields for extractive and abstractive QA  
- Explicit provenance and traceability to original datasets  
- Reproducibility through deterministic field mappings  

DoxplainQA is intentionally **model-agnostic** and **task-general**.

## Unified Schema

Each record in DoxplainQA conforms to the following schema:

| Field Name | Type | Description |
|----------|------|-------------|
| `dataset` | `str` | Name of the originating dataset |
| `id` | `str` | Original dataset-specific identifier |
| `question` | `str` | Natural language question |
| `answer` | `str` | Canonical answer string |
| `context` | `str` | Supporting textual context from which the answer is derived |

## Dataset Sources and Field Mappings

DoxplainQA currently integrates the following datasets:

- `boolq`
- `drop`
- `hotpotqa`
- `narrativeqa`
- `natural_questions`
- `qasper`
- `squad_v2`
- `triviaqa_wiki`

Each dataset is transformed independently into the unified schema using deterministic mappings documented below.

## Dataset-Specific Field Mappings

### 1. BoolQ

**Citation**  
Clark et al., *BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions*, NAACL 2019. [arXiv](https://arxiv.org/abs/1905.10044)

**Original Fields**

| BoolQ Field | Description |
|------------|-------------|
| `id` | Question identifier |
| `question` | Yes/no question |
| `passage` | Supporting passage |
| `answer` | Boolean label |

**Mapping to DoxplainQA**

| DoxplainQA Field | Source |
|------------------|--------|
| `dataset` | `"boolq"` |
| `id` | `id` |
| `question` | `question` |
| `answer` | Stringified boolean (`"yes"` / `"no"`) |
| `context` | `passage` |

### 2. DROP

**Citation**  
Dua et al., *DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs*, NAACL 2019. [arXiv](https://arxiv.org/abs/1903.00161)

**Original Fields**

| DROP Field | Description |
|-----------|-------------|
| `query_id` | Question identifier |
| `question` | Question text |
| `passage` | Passage text |
| `answers_spans` / `answers_number` | Answer annotations |

**Mapping to DoxplainQA**

| DoxplainQA Field | Source |
|------------------|--------|
| `dataset` | `"drop"` |
| `id` | `query_id` |
| `question` | `question` |
| `answer` | Normalized span or number answer |
| `context` | `passage` |

**Notes**
- Numerical and span answers are normalized to strings.
- Questions without resolvable answers are excluded.

### 3. HotpotQA

**Citation**  
Yang et al., *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*, EMNLP 2018. [arXiv](https://arxiv.org/abs/1809.09600)

**Original Fields**

| HotpotQA Field | Description |
|---------------|-------------|
| `_id` | Question identifier |
| `question` | Question text |
| `context` | Supporting paragraphs |
| `answer` | Answer string |

**Mapping to DoxplainQA**

| DoxplainQA Field | Source |
|------------------|--------|
| `dataset` | `"hotpotqa"` |
| `id` | `_id` |
| `question` | `question` |
| `answer` | `answer` |
| `context` | Concatenated paragraph texts |

### 4. NarrativeQA

**Citation**  
Kočiský et al., *The NarrativeQA Reading Comprehension Challenge*, TACL 2018. [arXiv](https://arxiv.org/abs/1712.07040)

**Original Fields**

| NarrativeQA Field | Description |
|------------------|-------------|
| `question_id` | Question identifier |
| `question` | Question text |
| `answer.text` | Human-generated answer |
| `summary` / `document` | Story context |

**Mapping to DoxplainQA**

| DoxplainQA Field | Source |
|------------------|--------|
| `dataset` | `"narrativeqa"` |
| `id` | `question_id` |
| `question` | `question` |
| `answer` | `answer.text` |
| `context` | Summary or full document text |

### 5. Natural Questions

**Citation**  
Kwiatkowski et al., *Natural Questions: A Benchmark for Question Answering Research*, TACL 2019. [ACL Anthology](https://aclanthology.org/Q19-1026/)

**Original Fields**

| NQ Field | Description |
|--------|-------------|
| `example_id` | Question identifier |
| `question_text` | Question |
| `document_text` | Wikipedia page |
| `short_answers` | Answer spans |

**Mapping to DoxplainQA**

| DoxplainQA Field | Source |
|------------------|--------|
| `dataset` | `"natural_questions"` |
| `id` | `example_id` |
| `question` | `question_text` |
| `answer` | Extracted short-answer text |
| `context` | `document_text` |

### 6. QASPER

**Citation**  
Dasigi et al., *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers*, NAACL 2021. [arXiv](https://arxiv.org/abs/2105.03011)

**Original Fields**

| QASPER Field | Description |
|-------------|-------------|
| `question_id` | Question identifier |
| `question` | Question text |
| `evidence` | Supporting sections |
| `answer.answer_text` | Free-form answer |

**Mapping to DoxplainQA**

| DoxplainQA Field | Source |
|------------------|--------|
| `dataset` | `"qasper"` |
| `id` | `question_id` |
| `question` | `question` |
| `answer` | `answer.answer_text` |
| `context` | Concatenated evidence text |

### 7. SQuAD v2.0

**Citation**  
Rajpurkar et al., *Know What You Don’t Know: Unanswerable Questions for SQuAD*, ACL 2018. [arXiv](https://arxiv.org/abs/1806.03822)

**Original Fields**

| SQuAD Field | Description |
|------------|-------------|
| `id` | Question identifier |
| `question` | Question text |
| `context` | Paragraph |
| `answers.text` | Answer spans |

**Mapping to DoxplainQA**

| DoxplainQA Field | Source |
|------------------|--------|
| `dataset` | `"squad_v2"` |
| `id` | `id` |
| `question` | `question` |
| `answer` | Answer span text (or empty for unanswerable) |
| `context` | `context` |

### 8. TriviaQA (Wikipedia)

**Citation**  
Joshi et al., *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension*, ACL 2017. [arXiv](https://arxiv.org/abs/1705.03551)

**Original Fields**

| TriviaQA Field | Description |
|---------------|-------------|
| `question_id` | Question identifier |
| `question` | Trivia question |
| `answer.value` | Answer string |
| `entity_pages[].wiki_context` | Wikipedia context |

**Mapping to DoxplainQA**

| DoxplainQA Field | Source |
|------------------|--------|
| `dataset` | `"triviaqa_wiki"` |
| `id` | `question_id` |
| `question` | `question` |
| `answer` | `answer.value` |
| `context` | Concatenated Wikipedia contexts |

## Design Rationale

The DoxplainQA schema is intentionally minimal. All task-specific or structural information not expressible through the five core fields is removed to ensure:

- Consistent model interfaces  
- Simplified evaluation logic  
- Cross-dataset comparability  

This design prioritizes *practical interoperability* over dataset completeness.

## Licensing

**Each dataset retains its original license**. Users must comply with the individual licensing terms of:

- BoolQ (CC BY-SA 3.0)  
- DROP (CC BY-SA 4.0)  
- HotpotQA (CC BY-SA 4.0)  
- NarrativeQA (CC BY 4.0)  
- Natural Questions (CC BY-SA 3.0)  
- QASPER (CC BY 4.0)  
- SQuAD v2.0 (CC BY-SA 4.0)  
- TriviaQA (Apache 2.0)

DoxplainQA introduces no additional licensing terms.

## Citation

Please **cite the original datasets** in all research conducted with DoxplainQA. If you would like to cite this repository, consider the details below.

```bibtex
@misc{doxplainqa,
  title  = {DoxplainQA: A Unified Question--Answering Dataset},
  author = {Van Dyke, Katelyn},
  year   = {2025},
  note   = {Unified schema aggregation of eight QA benchmarks}
}
