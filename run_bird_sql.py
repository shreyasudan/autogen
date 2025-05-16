import os
import json
import sqlite3
import argparse
import asyncio
import pandas as pd
import logging
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Tuple

# Set environment variables for proper Unicode handling
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bird_sql_autogen.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# AutoGen imports
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

class BirdSQLProcessor:
    """Main processor for the BIRD-SQL Mini-Dev benchmark"""
    
    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        dataset_path: str = "./minidev/MINIDEV/mini_dev_sqlite.json",
        db_dir: str = "./minidev/MINIDEV/dev_databases",
        output_dir: str = "./results",
        api_key: Optional[str] = None,
    ):
        """Initialize the BIRD-SQL Processor
        
        Args:
            model_name: Name of the OpenAI model to use
            dataset_path: Path to the BIRD-SQL Mini-Dev dataset JSON
            db_dir: Directory containing the databases
            output_dir: Directory to save results
            api_key: OpenAI API key (optional, will use env var if not provided)
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.db_dir = db_dir
        self.output_dir = output_dir
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        # Initialize the model client
        self.model_client = OpenAIChatCompletionClient(
            model=model_name,
            api_key=self.api_key,
            default_headers={"Content-Type": "application/json; charset=utf-8"}
        )
        
        # Initialize agents and group chat
        self._initialize_agents()
        
        # Track accumulated knowledge
        self.knowledge_base = {}
    
    def _find_dataset_path(self) -> str:
        """Find the dataset path by checking common locations"""
        possible_paths = [
            "./minidev/MINIDEV/mini_dev_sqlite.json",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found dataset at {path}")
                return path
        
        # If no path is found, use the most likely path and let it fail explicitly later
        logger.warning("Could not find dataset file. Using default path.")
        return "./minidev/MINIDEV/mini_dev_sqlite.json"
    
    def _find_db_dir(self) -> str:
        """Find the database directory by checking common locations"""
        possible_paths = [
            "./mini_dev_data/dev_databases",
            "./minidev/MINIDEV/dev_databases",
            "./MINIDEV/dev_databases",
            "./dev_databases"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found database directory at {path}")
                return path
        
        # If no path is found, use the most likely path and let it fail explicitly later
        logger.warning("Could not find database directory. Using default path.")
        return "./minidev/MINIDEV/dev_databases"
        
    def _initialize_agents(self):
        """Initialize all agents for the multi-agent system"""
        
        # Schema Analyzer Agent
        self.schema_analyzer = AssistantAgent(
            name="SchemaAnalyzer",
            system_message="""You are a database schema expert. Your role is to analyze database schemas 
            and identify the relevant tables and columns needed to answer a question. 
            You should identify:
            1. The primary tables needed
            2. Any join relationships required
            3. The specific columns that contain the information needed
            4. Any potential filters or conditions
            
            Provide your analysis in a structured format with clear reasoning.
            """,
            model_client=self.model_client
        )
        
        # SQL Generator Agent
        self.sql_generator = AssistantAgent(
            name="SQLGenerator",
            system_message="""You are a SQL expert specializing in generating precise SQL queries based on
            natural language questions and database schemas. Your task is to:
            1. Understand the question thoroughly
            2. Review the schema analysis provided
            3. Generate a correct SQL query that answers the question
            4. Explain your query construction reasoning
            
            Focus on correctness. Show your reasoning step-by-step before providing the final SQL query.
            The final SQL query should be enclosed within ```sql and ``` tags.
            """,
            model_client=self.model_client
        )
        
        # SQL Optimizer Agent
        self.sql_optimizer = AssistantAgent(
            name="SQLOptimizer",
            system_message="""You are a SQL optimization expert. Your role is to review and optimize
            SQL queries for both correctness and efficiency. You should:
            1. Verify query correctness
            2. Ensure all joins have proper conditions
            3. Optimize query structure (use proper indexing hints if appropriate)
            4. Simplify complex expressions
            5. Use more efficient alternatives where possible
            
            Return the optimized SQL query within ```sql and ``` tags. If no changes are needed,
            state why the original query is already optimal.
            """,
            model_client=self.model_client
        )
        
        # Validator Agent
        self.validator = AssistantAgent(
            name="Validator",
            system_message="""You are a SQL validator. Your task is to review generated SQL queries
            and identify potential issues including:
            1. Syntax errors
            2. Incorrect table or column references
            3. Missing joins or join conditions
            4. Logical errors in conditions
            5. Potential execution errors
            
            If you identify issues, explain them clearly and suggest corrections.
            """,
            model_client=self.model_client
        )
        
        # User Proxy Agent - capable of executing code and database operations
        self.group_chat = RoundRobinGroupChat(
        participants=[self.schema_analyzer, self.sql_generator, self.sql_optimizer, self.validator],
        termination_condition=TextMentionTermination("TASK_COMPLETE", sources=["Validator"])
   )
    
    async def process_query(self, query_data: Dict[str, Any]) -> str:
        """Process a single query from the dataset
        
        Args:
            query_data: Query data from the dataset
            
        Returns:
            Generated SQL query
        """
        db_id = query_data["db_id"]
        question = query_data["question"]
        evidence = query_data.get("evidence", "")
        
        # Get database schema
        db_schema = self._get_db_schema(db_id)
        
        # Prepare context with similar examples if available
        context = self._get_knowledge_context(db_id, question)
        
        # Create task prompt
        task_prompt = f"""
        # Database: {db_id}
        
        ## Question:
        {question}
        
        ## Evidence:
        {evidence}
        
        ## Database Schema:
        {db_schema}
        
        {context}
        
        Generate a SQL query that answers this question.
        """
        
        logger.info(f"Processing question: {question}")
        
        # Run the multi-agent conversation to generate SQL
        chat_result = await self.group_chat.run(task=task_prompt)
        # chat_result = ""
        # async for chunk in self.group_chat.run_stream(task=task_prompt):
        #     # `chunk` is usually a delta object with `.content` or `.message`
        #     chat_result += chunk.content  
        
        try:
            # Extract the SQL from the conversation
            sql = self._extract_sql_from_result(chat_result)
            
            # Validate the SQL
            sql = await self._validate_sql(db_id, sql)
            
            # Store the successful query in the knowledge base
            self._update_knowledge_base(db_id, question, sql)
            
            return sql
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return ""
    
    def _get_db_schema(self, db_id: str) -> str:
        """Get the database schema for a given database ID
        
        Args:
            db_id: Database ID
            
        Returns:
            Formatted schema information
        """
        db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            tables = [table[0] for table in tables]
            
            schema_info = []
            
            # Get schema for each table
            for table in tables:
                cursor.execute(f'PRAGMA table_info("{table}");')
                columns = cursor.fetchall()
                column_info = [f"  - {col[1]}: {col[2]}" for col in columns]
                
                # Get sample data for this table (first 3 rows)
                cursor.execute(f'SELECT * FROM "{table}" LIMIT 3;')
                sample_data = cursor.fetchall()
                
                samples = []
                if sample_data:
                    sample_headers = [col[1] for col in columns]
                    for row in sample_data:
                        samples.append(", ".join([f"{h}: {v}" for h, v in zip(sample_headers, row)]))
                
                schema_info.append(f"Table: {table}")
                schema_info.append("Columns:")
                schema_info.extend(column_info)
                
                if samples:
                    schema_info.append("Sample data:")
                    schema_info.extend([f"  - {s}" for s in samples])
                
                schema_info.append("")
            
            conn.close()
            return "\n".join(schema_info)
            
        except Exception as e:
            logger.error(f"Error getting schema for {db_id}: {e}")
            return f"Error getting schema: {e}"
    
    def _get_knowledge_context(self, db_id: str, question: str) -> str:
        """Get similar examples from the knowledge base
        
        Args:
            db_id: Database ID
            question: The current question
            
        Returns:
            Context with similar examples
        """
        if db_id not in self.knowledge_base:
            return ""
        
        examples = self.knowledge_base[db_id]
        
        # For simplicity, just return the last 3 examples for the same database
        # In a real implementation, you would use embedding similarity to find similar examples
        if len(examples) > 0:
            context = ["## Similar Examples:"]
            
            for i, example in enumerate(examples[-3:]):
                context.append(f"### Example {i+1}:")
                context.append(f"Question: {example['question']}")
                context.append(f"SQL: {example['sql']}")
                context.append("")
            
            return "\n".join(context)
        else:
            return ""
    
    def _extract_sql_from_result(self, result: str) -> str:
        """Extract SQL from the conversation result
        
        Args:
            result: Conversation result
            
        Returns:
            Extracted SQL query
        """
        # Look for SQL enclosed in ```sql ... ``` tags
        import re
        sql_pattern = r"```sql\s*(.*?)\s*```"
        matches = re.findall(sql_pattern, result, re.DOTALL)
        
        if matches:
            return matches[-1].strip()  # Return the last SQL block
        else:
            raise ValueError("No SQL found in result")
    
    async def _validate_sql(self, db_id: str, sql: str) -> str:
        """Validate the SQL by trying to execute it
        
        Args:
            db_id: Database ID
            sql: SQL query to validate
            
        Returns:
            Validated SQL query
        """
        db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            conn.close()
            return sql
        except Exception as e:
            logger.warning(f"SQL validation error: {e}")
            
            # Ask the validator to fix the SQL
            task = f"""
            The SQL query has an error: {e}
            
            Original SQL:
            ```sql
            {sql}
            ```
            
            Please fix this SQL query.
            """
            
            result = await self.validator.run(task=task)
            
            # Extract the fixed SQL
            fixed_sql = self._extract_sql_from_result(result)
            
            # Validate the fixed SQL
            return await self._validate_sql(db_id, fixed_sql)
    
    def _update_knowledge_base(self, db_id: str, question: str, sql: str):
        """Update the knowledge base with a successful query
        
        Args:
            db_id: Database ID
            question: Question
            sql: SQL query
        """
        if db_id not in self.knowledge_base:
            self.knowledge_base[db_id] = []
        
        self.knowledge_base[db_id].append({
            "question": question,
            "sql": sql
        })
    
    async def process_all(self) -> List[Dict[str, Any]]:
        """Process all queries in the dataset
        
        Returns:
            List of results for each query
        """
        results = []
        
        for example in tqdm(self.dataset, desc="Processing queries"):
            try:
                sql = await self.process_query(example)
                results.append({
                    "db_id": example["db_id"],
                    "question": example["question"],
                    "sql": sql
                })
            except Exception as e:
                logger.error(f"Error processing example: {e}")
                results.append({
                    "db_id": example["db_id"],
                    "question": example["question"],
                    "sql": "",
                    "error": str(e)
                })
        
        # Save results
        output_path = os.path.join(self.output_dir, f"predictions_{self.model_name.replace('-', '_')}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save BIRD-SQL file in binary UTF-8:
        bird_format = []
        for res in results:
            sql = res["sql"].replace("\u2018", "'").replace("\u2019", "'")
            bird_format.append(f"{sql}\t----- bird -----\t{res['db_id']}")
        bird_bytes = "\n".join(bird_format).encode("utf-8")
        bird_output_path = os.path.join(self.output_dir, f"predict_mini_dev_{self.model_name.replace('-', '_')}.sql")
        with open(bird_output_path, "wb") as f:
            f.write(bird_bytes)
        
        return results
    
    async def close(self):
        """Close the model client"""
        await self.model_client.close()

async def main():
    parser = argparse.ArgumentParser(description="BIRD-SQL Mini-Dev Processor with AutoGen")
    parser.add_argument("--model-name", type=str, default="gpt-4-turbo", help="OpenAI model to use")
    parser.add_argument("--dataset-path", type=str, default="./minidev/MINIDEV/mini_dev_sqlite.json", help="Path to dataset")
    parser.add_argument("--db-dir", type=str, default="./minidev/MINIDEV/dev_databases", help="Directory with databases")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API key (optional)")
    
    args = parser.parse_args()
    
    processor = BirdSQLProcessor(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        db_dir=args.db_dir,
        output_dir=args.output_dir,
        api_key=args.api_key
    )
    
    try:
        results = await processor.process_all()
        print(f"Processed {len(results)} queries. Results saved to {args.output_dir}")
    finally:
        await processor.close()

if __name__ == "__main__":
    asyncio.run(main())