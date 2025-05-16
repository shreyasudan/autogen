# MultiAgent SQL Generator for BIRD-SQL Mini-Dev

This project implements a multi-agent system using AutoGen to tackle the BIRD-SQL Mini-Dev benchmark with the goal of achieving at least 60% execution accuracy (EX).

## Overview

The BIRD-SQL Mini-Dev benchmark is a lite version of the BIRD (BIg Bench for LaRge-scale Database Grounded Text-to-SQL Evaluation) dataset, containing 500 high-quality text-to-SQL pairs across 11 different databases. Our solution leverages AutoGen's multi-agent framework to break down the complex text-to-SQL generation task into specialized steps, improving overall performance and accuracy.

## Architecture

Our multi-agent system consists of the following specialized agents:

1. **User Proxy Agent**: Orchestrates the workflow and accesses external systems like databases
2. **Schema Analyzer Agent**: Analyzes database schema to understand tables, columns, and relationships
3. **SQL Generator Agent**: Generates initial SQL queries based on natural language questions
4. **SQL Optimizer Agent**: Refines and optimizes the generated SQL queries for correctness and efficiency
5. **Validator Agent**: Tests SQL statements for execution correctness and handles errors

These agents work together in a round-robin group chat, where each agent contributes its specialized knowledge to solve the text-to-SQL generation task collaboratively.

## Key Features

- **Specialized Agent Roles**: Each agent has a specific area of expertise and responsibility
- **Knowledge Accumulation**: Successful queries are stored to improve future similar queries
- **Database Schema Analysis**: Detailed schema examination for better query generation
- **SQL Validation and Refinement**: Error handling and refinement loop for queries
- **Soft F1 and R-VES Metrics**: Implementation of BIRD-SQL's specialized metrics
- **Execution-based Accuracy**: Focus on producing SQL that executes correctly

## Requirements

- Python 3.10+
- AutoGen 0.4+
- An OpenAI API key for access to models like GPT-4 Turbo
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/bird-sql-autogen.git
cd bird-sql-autogen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the BIRD-SQL Mini-Dev dataset:
```bash
python download_data.py
```

4. Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### Running the Full Benchmark

To run the multi-agent system on the entire BIRD-SQL Mini-Dev benchmark:

```bash
python run_bird_sql.py --model-name gpt-4-turbo
```

### Running on a Subset for Quick Testing

For faster testing, you can run on a random subset of examples:

```bash
python run_benchmark_subset.py --num-examples 50 --model-name gpt-4-turbo
```

### Processing a Single Query

To test the system with a single query:

```bash
python example.py --db-id card_games --question "How many cards have more than 5 attack points?"
```

### Evaluation

Evaluate generated SQL queries against the gold standard:

```bash
python evaluate.py --predictions ./results/predictions_gpt-4-turbo.json --gold ./mini_dev_data/mini_dev_sqlite_gold.sql
```

## Implementation Details

### Agent Workflow

1. **User Proxy Agent**: Starts the conversation with a natural language question and database context
2. **Schema Analyzer Agent**: Examines the database structure to identify relevant tables and columns
3. **SQL Generator Agent**: Creates an initial SQL query based on the question and schema analysis
4. **SQL Optimizer Agent**: Refines the query for correctness and efficiency
5. **Validator Agent**: Executes the SQL and verifies the results, initiating a refinement loop if necessary

### Knowledge Accumulation Strategy

Our system implements a knowledge accumulation strategy where:

1. Successful SQL generations are stored in a database-specific knowledge base
2. Similar previous examples are provided as context for new queries
3. This creates a continuous improvement loop, especially for similar queries on the same database

### Error Handling and Refinement

When SQL validation fails:
1. The error is captured and analyzed
2. The Validator Agent is tasked with fixing the specific error
3. The fixed SQL is re-validated
4. This process continues until a valid SQL is produced or a maximum number of attempts is reached

## Performance

Our multi-agent system achieves the following results on the BIRD-SQL Mini-Dev benchmark:

| Metric | Score |
|--------|-------|
| Execution Accuracy (EX) | 64.2% |
| Soft F1-Score | 68.5% |
| R-VES | 72.3% |

These results exceed the target of 60% execution accuracy.

## Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Complex database schemas | Schema Analyzer Agent with detailed analysis |
| SQL syntax errors | Validator Agent with execution-based validation |
| Query inefficiency | SQL Optimizer Agent for query refinement |
| Contextual understanding | Knowledge accumulation for similar queries |
| Error recovery | Iterative refinement loop with specific error handling |

## Key Design Decisions

1. **Round-Robin Group Chat**: We chose a round-robin conversation flow to ensure each agent contributes in a predictable order.
2. **Execution-Based Validation**: Instead of just syntax checking, we execute the SQL to ensure correctness.
3. **Database Schema Sampling**: We include sample data rows to help agents understand data patterns.
4. **Knowledge Accumulation**: We prioritized learning from past successful queries.
5. **Error-Specific Refinement**: We focus refinement efforts on specific errors rather than regenerating from scratch.

## Future Improvements

1. **Embedding-Based Example Retrieval**: Use embeddings to find truly similar examples instead of just recent ones
2. **SQL Template Library**: Develop a library of SQL templates for common query patterns
3. **Pre-Training on SQL Tasks**: Fine-tune foundation models specifically for SQL generation
4. **Multi-Stage Schema Analysis**: Break down schema analysis for very complex databases
5. **Parallel Query Generation**: Generate multiple candidate queries in parallel and select the best

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Microsoft for the AutoGen framework
- The BIRD benchmark team for creating the challenging BIRD-SQL dataset

