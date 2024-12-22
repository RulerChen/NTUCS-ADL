import copy
import gc
import json
import os
import random
import re
import sqlite3
import warnings

import torch
from colorama import Fore, Style
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import logging as transformers_logging

from base import Agent
from utils import (
    RAG,
    get_fewshot_COT_template,
    get_system_prompt_onlyone_ans,
    get_zeroshot_COT_prompt,
    strip_all_lines,
)

# Ignore warning messages from transformers
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

filter_system_prompt = """
You are an SQL expert.  
You will be given a database schema and a user query and your task is to filter the database schema to only include the relevant tables that are required to answer the user query.
"""

filter_user_prompt_oneshot = """
You are tasked with filtering a given database schema to return only the relevant tables that are required to answer the user query.
A table is considered relevant if it is referenced directly or indirectly through foreign key relationships in the user query.
Note that you don't need to give me any explanation. Just provide the filtered table schema.

The following example is for your reference:
**Schema:**  
CREATE TABLE employees (  
    employee_id INTEGER NOT NULL PRIMARY KEY, 
    name TEXT NOT NULL, 
    department_id INTEGER NOT NULL, 
    FOREIGN KEY (department_id) REFERENCES departments(department_id)  
)

CREATE TABLE departments (  
    department_id INTEGER NOT NULL PRIMARY KEY,
    department_name TEXT NOT NULL 
)

CREATE TABLE projects (  
    project_id INTEGER NOT NULL PRIMARY KEY, 
    project_name TEXT NOT NULL 
)

**User Query:**  
Retrieve the names of employees and their departments.

**Filtered Table:**  
- employees
- departments

===============

Now, it's your turn.

**Schema:**
{table_schema}

**User Query:**
{user_query}

Please remove the irrelevant tables and return the filtered table schema in the following format:
- <table_name_1>
- <table_name_2>
- ...
"""

filter_user_prompt_fewshot = """
You are tasked with filtering a given database schema to return only the relevant tables that are required to answer the user query.
A table is considered relevant if it is referenced directly or indirectly through foreign key relationships in the user query.
Note that you don't need to give me any explanation. Just provide the filtered table schema.

The following examples are for your reference:
{fewshot_examples}

===============

Now, it's your turn.

**Schema:**
{table_schema}

**User Query:**
{user_query}

Please remove the irrelevant tables and return the filtered table schema in the following format:
- <table_name_1>
- <table_name_2>
- ...
"""

generate_system_prompt = """
You are an SQLite expert.
Your task is to generate accurate SQL queries based on the given table schema and user query.
"""

generate_user_prompt_oneshot = """
You are tasked with generating the correct SQLite SQL code to answer the user query based on the given database schema.

The following example is for your reference:
**Schema:**
CREATE TABLE employees (
    employee_id INTEGER NOT NULL PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
)

CREATE TABLE departments (
    department_id INTEGER NOT NULL PRIMARY KEY,
    department_name TEXT NOT NULL
)

CREATE TABLE projects (
    project_id INTEGER NOT NULL PRIMARY KEY,
    project_name TEXT NOT NULL
)

**User Query:**
Retrieve the names of employees and their departments.

**Answer:**
```sql
SELECT employees.name, departments.department_name
FROM employees
JOIN departments ON employees.department_id = departments.department_id
```

===============

Now, it's your turn.

**Schema:**
{table_schema}

**User Query:**
{user_query}

Please generate the correct SQLite SQL to answer the user query in the following format:
```sql\n<your_SQL_code>\n```

You don't need to explain your reasoning for the SQL. Just provide the SQL starting with ```sql and ending with ```.
"""

generate_user_prompt_fewshot = """
You are tasked with generating the correct SQLite SQL code to answer the user query based on the given database schema.

The following examples are for your reference:
{fewshot_examples}

===============

Now, it's your turn.

**Schema:**
{table_schema}

**User Query:**
{user_query}

Please generate the correct SQLite SQL code to answer the user query in the following format:
```sql\n<your_SQL_code>\n```

You don't need to explain your reasoning for the SQL. Just provide the SQL starting with ```sql and ending with ```.
"""

error_message = """

**Error:**
Seems like you don't follow the format correctly previously.
Please make sure to provide the SQL code in the following format: 
```sql\n<your_SQL_code>\n```
"""

sql_error_message = """

**Error:**
The SQL code you provided previously contains errors. 
Below is the error message returned by SQLite. Please revise the SQL code accordingly to fix the issue.

Your Previous SQL Code:
{sql_code}

SQLite Error Message:
{message}

Guidelines for Fixing SQL Code:
1. Check if the referenced columns and tables exist in the schema. Correct any misspelled or missing names.
2. If using aggregate functions (e.g., COUNT, AVG), ensure proper usage (e.g., using GROUP BY if necessary).
3. SQLite does not support some functions (e.g., YEAR). Use alternative methods or rewrite the query.
4. Avoid incomplete input or invalid syntax (e.g., ensure all clauses like SELECT, WHERE, and GROUP BY are properly formed).
5. Verify the logic aligns with the intended query. Provide accurate SQL based on the schema.

Please revise your SQL code based on these guidelines and errors to generate a valid SQL query in the following format:
```sql\n<your_SQL_code>\n```

You don't need to explain your reasons. Just provide the SQL starting with ```sql and ending with ```.
"""


class SQLGenerationAgent(Agent):
    """
    An agent that generates SQL code based on the given table schema and the user query.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize your LLM here
        """
        super().__init__(config)
        self.llm_config = config
        if config["use_8bit"]:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_has_fp16_weight=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name1"],
                quantization_config=quantization_config,
                device_map=config["device"],
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name1"],
                torch_dtype=torch.float16,
                device_map=config["device"],
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name1"])
        self.model.eval()
        self.count = 0

        self.model_name = config["model_name1"]
        self.user_query_list = []
        self.table_schema_list = []
        self.table_schema_filter_list = []
        self.table_name_list = []
        self.sql_code_list = []
        self.filter_prompt_list = []
        self.generate_prompt_list = []
        self.error = False
        self.error_message_list = []
        self.rag = RAG(config["rag"])

    def change_model(self) -> None:
        if hasattr(self, "model"):
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            gc.collect()
        self.model_name = (
            self.llm_config["model_name1"]
            if self.model_name == self.llm_config["model_name2"]
            else self.llm_config["model_name2"]
        )
        print(Fore.GREEN + f"Change model to {self.model_name}" + Style.RESET_ALL)

        if self.llm_config["use_8bit"]:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_has_fp16_weight=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=llm_config["device"],
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=llm_config["device"],
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()

    def __call__(self, table_schema: str, user_query: str) -> str:
        self.reset_log_info()
        self.count += 1

        if self.count % 10 == 0:
            self.change_model()
            self.count = 0

        pred_text = self.get_filtered_candidates(table_schema, user_query)

        table_names = self.extract_table_names(pred_text)
        table_schema_filter = self.extract_schema(table_schema, table_names)

        sql_code = self.generate_sql_code(table_schema_filter, table_schema, user_query)

        self.user_query_list.append(user_query)
        self.table_schema_list.append(table_schema)
        self.table_schema_filter_list.append(table_schema_filter)
        self.table_name_list.append(table_names)
        self.sql_code_list.append(sql_code)

        return sql_code

    def update(self, correctness: bool) -> bool:
        # if self.error:
        #     with open("./log/error_log.txt", "a") as f:
        #         f.write("\n" + "-" * 20 + "\n")
        #         f.write(f"Error message:\n{self.error_message_list[-1]}\n\n")
        #         f.write(f"User query:\n{self.user_query_list[-1]}\n\n")
        #         f.write(f"Table name:\n{self.table_name_list[-1]}\n\n")
        #         f.write(f"Table schema:\n{self.table_schema_list[-1]}\n\n")
        #         f.write(f"Table schema filter:\n{self.table_schema_filter_list[-1]}\n\n")
        #         f.write(f"SQL code:\n{self.sql_code_list[-1]}\n\n")
        #         f.write(f"Filter prompt:\n{self.filter_prompt_list[-1]}\n\n")
        #         f.write(f"Generate prompt:\n{self.generate_prompt_list[-1]}\n")
        #         f.write("-" * 20 + "\n")

        torch.cuda.empty_cache()
        gc.collect()

        if correctness:
            user_query = self.user_query_list[-1]
            table_name = self.table_name_list[-1]
            table_schema_filter = self.table_schema_filter_list[-1]
            sql_code = self.sql_code_list[-1]

            chunk = {
                "schema": table_schema_filter,
                "query": user_query,
                "sql": sql_code,
                "filter_table": "\n".join(f"- {name}" for name in table_name),
            }
            chunk = json.dumps(chunk)
            self.rag.insert(key=user_query, value=chunk, schema=table_schema_filter)
            return True

        self.user_query_list = []
        self.table_schema_list = []
        self.table_schema_filter_list = []
        self.table_name_list = []
        self.sql_code_list = []
        self.filter_prompt_list = []
        self.generate_prompt_list = []
        self.error = False
        self.error_message_list = []

        return False

    def get_filtered_candidates(self, table_schema: str, user_query: str) -> str:
        shots = (
            self.rag.retrieve(query=user_query, top_k=self.rag.top_k, schema=table_schema)
            if (self.rag.insert_acc > 0)
            else []
        )

        if len(shots) > 0:
            shots = [json.loads(shot) for shot in shots]
            fewshot_examples = "\n\n".join(
                [
                    f"**Example {i + 1}:**\n**Schema:**\n{shot['schema']}\n\n**User Query:**\n{shot['query']}\n\n**Filtered Table:**\n{shot['filter_table']}"
                    for i, shot in enumerate(shots)
                ]
            )
            messages = [
                {"role": "system", "content": filter_system_prompt},
                {
                    "role": "user",
                    "content": filter_user_prompt_fewshot.format(
                        fewshot_examples=fewshot_examples,
                        table_schema=table_schema,
                        user_query=user_query,
                    ),
                },
            ]
        else:
            messages = [
                {"role": "system", "content": filter_system_prompt},
                {
                    "role": "user",
                    "content": filter_user_prompt_oneshot.format(
                        table_schema=table_schema, user_query=user_query
                    ),
                },
            ]

        pred_text = self.generate_response(messages)
        self.filter_prompt_list.append(messages[1]["content"])

        return pred_text

    def extract_table_names(self, pred_text: str) -> list:
        table_names = re.findall(r"^\s*-\s*(\w+)", pred_text, re.MULTILINE)
        return table_names

    def extract_schema(self, table_schema: str, tables: list) -> str:
        pattern = re.compile(r'CREATE TABLE\s+[`"\']?(\w+)[`"\']?.*?(?=\n\n|$)', re.DOTALL)
        selected_tables = [
            match.group(0)
            for match in re.finditer(pattern, table_schema)
            if match.group(1) in tables
        ]

        return "\n\n".join(selected_tables)

    def generate_sql_code(
        self, table_schema_filter: str, table_schema: str, user_query: str
    ) -> str:
        shots = (
            self.rag.retrieve(query=user_query, top_k=self.rag.top_k, schema=table_schema_filter)
            if (self.rag.insert_acc > 0)
            else []
        )

        if len(shots) > 0:
            shots = [json.loads(shot) for shot in shots]
            fewshot_examples = "\n\n".join(
                [
                    f"**Example {i + 1}:**\n**Schema:**\n{shot['schema']}\n\n**User Query:**\n{shot['query']}\n\n**SQL Code:**\n```sql\n{shot['sql']}\n```"
                    for i, shot in enumerate(shots)
                ]
            )
            messages = [
                {"role": "system", "content": generate_system_prompt},
                {
                    "role": "user",
                    "content": generate_user_prompt_fewshot.format(
                        fewshot_examples=fewshot_examples,
                        table_schema=table_schema_filter,
                        user_query=user_query,
                    ),
                },
            ]
        else:
            messages = [
                {"role": "system", "content": generate_system_prompt},
                {
                    "role": "user",
                    "content": generate_user_prompt_oneshot.format(
                        table_schema=table_schema_filter, user_query=user_query
                    ),
                },
            ]

        count = 0
        message = copy.deepcopy(messages)

        while count < 5:
            count += 1

            pred_text = self.generate_response(message)
            sql_code, correct = self.parse_sql(pred_text)

            if not correct:
                message = copy.deepcopy(messages)
                message[1]["content"] += error_message
                continue
            else:
                message[1]["content"] = messages[1]["content"]

            syntax_valid, syntax_message = self.validate_sql_syntax(sql_code, table_schema)
            if not syntax_valid:
                message = copy.deepcopy(messages)
                message[1]["content"] += sql_error_message.format(
                    sql_code=sql_code, message=syntax_message
                )
                self.generate_prompt_list.append(message[1]["content"])
                if count % 2 == 0:
                    self.change_model()
                    self.count = 0
                continue
            else:
                self.generate_prompt_list.append(message[1]["content"])
                break

        return sql_code

    def validate_sql_syntax(self, sql_code: str, table_schema: str) -> tuple:
        try:
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()

            schema_statements = [
                statement.strip() for statement in table_schema.split("\n\n") if statement.strip()
            ]

            for statement in schema_statements:
                if statement == "CREATE TABLE sqlite_sequence(name,seq)":
                    continue
                if statement.startswith(")"):
                    continue
                if statement.startswith("CREATE TABLE constructorResults"):
                    statement += "\n)"
                cursor.execute(statement)

            sql_code = sql_code.replace(";", "").strip()
            cursor.execute(sql_code)
            conn.close()
            return True, "Valid SQL syntax"

        except sqlite3.Error as e:
            print(Fore.RED + f"SQLite syntax error: {e}" + Style.RESET_ALL)
            self.error = True
            self.error_message_list.append(f"SQLite syntax error: {e}")
            return False, f"SQLite syntax error: {e}"

    def generate_response(self, messages: list) -> str:
        """
        Generate a response using the local model.
        """
        text_chat = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_chat], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs, max_new_tokens=self.llm_config["max_tokens"], do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def parse_sql(self, pred_text: str) -> dict:
        pattern = r"```sql([\s\S]*?)```"
        match = re.search(pattern, pred_text)
        if match:
            sql_code = match.group(1)
            sql_code = sql_code.strip()
            return sql_code, True
        else:
            start_pattern = r"```sql([\s\S]*)"
            start_match = re.search(start_pattern, pred_text)
            if start_match:
                sql_code = start_match.group(1).strip()
                print(
                    Fore.YELLOW
                    + "Warning: SQL code block is not properly closed."
                    + Style.RESET_ALL
                )
                return sql_code, True
            else:
                sql_code = pred_text
                print(Fore.RED + "SQL code format is incorrect." + Style.RESET_ALL)
        return sql_code, False


class LocalModelAgent(Agent):
    """
    A base agent that uses a local model for text generation tasks.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the local model
        """
        super().__init__(config)
        self.llm_config = config
        if config["use_8bit"]:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_has_fp16_weight=False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name1"],
                quantization_config=quantization_config,
                device_map=config["device"],
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config["model_name1"], torch_dtype=torch.float16, device_map=config["device"]
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name1"])
        self.rag = RAG(config["rag"])
        # Save the streaming inputs and outputs for iterative improvement
        self.inputs = list()
        self.self_outputs = list()
        self.responses = list()
        self.model.eval()

    def generate_response(self, messages: list) -> str:
        """
        Generate a response using the local model.
        """
        text_chat = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_chat], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=self.llm_config["max_tokens"], do_sample=False
            )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        torch.cuda.empty_cache()
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def update(self, correctness: bool) -> bool:
        """
        Update the agent based on the correctness of its output.
        """
        if correctness:
            question = self.inputs[-1]
            # answer = self.self_outputs[-1]
            answer = self.responses[-1]
            chunk = self.get_shot_template().format(question=question, answer=answer)
            self.rag.insert(key=question, value=chunk)
            return True
        return False


class ClassificationAgent(LocalModelAgent):
    """
    An agent that classifies text into one of the labels in the given label set.
    """

    @staticmethod
    def get_shot_template() -> str:
        prompt = f"""\
        {{question}}
        Output: {{answer}}"""
        return strip_all_lines(prompt)

    def __call__(self, label2desc: dict[str, str], text: str) -> str:
        self.reset_log_info()
        # print(text)

        option_text = "\n".join([f"{str(k)}. {v}" for k, v in label2desc.items()])
        system_prompt = get_system_prompt_onlyone_ans()
        prompt_zeroshot = get_zeroshot_COT_prompt(option_text, text)
        prompt_fewshot = get_fewshot_COT_template(option_text, text)

        shots = (
            self.rag.retrieve(query=text, top_k=self.rag.top_k) if (self.rag.insert_acc > 0) else []
        )
        if len(shots):
            fewshot_text = "\n\n\n".join(shots).replace("\\", "\\\\")
            try:
                prompt = re.sub(
                    pattern=r"\{fewshot_text\}", repl=fewshot_text, string=prompt_fewshot
                )
            except Exception as e:
                error_msg = f"Error ```{e}``` caused by these shots. Using the zero-shot prompt."
                print(Fore.RED + error_msg + Fore.RESET)
                prompt = prompt_zeroshot
        else:
            print(Fore.YELLOW + "No RAG shots found. Using zeroshot prompt." + Fore.RESET)
            prompt = prompt_zeroshot

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self.generate_response(messages)
        print(response)
        prediction = self.extract_label(response, label2desc)

        self.update_log_info(
            log_data={
                "num_input_tokens": len(self.tokenizer.encode(system_prompt + prompt)),
                "num_output_tokens": len(self.tokenizer.encode(response)),
                "num_shots": str(len(shots)),
                "input_pred": prompt,
                "output_pred": response,
            }
        )
        self.inputs.append(text)
        self.responses.append(response)
        self.self_outputs.append(f"{str(prediction)}. {label2desc[int(prediction)]}")
        return prediction

    @staticmethod
    def extract_label(pred_text: str, label2desc: dict[str, str]) -> str:
        numbers = re.findall(pattern=r"(\d+)\.", string=pred_text)
        if len(numbers) == 1:
            number = numbers[0]
            if int(number) in label2desc:
                prediction = number
            else:
                print(
                    Fore.RED
                    + f"Prediction {pred_text} not found in the label set. Randomly select one."
                    + Style.RESET_ALL
                )
                prediction = random.choice(list(label2desc.keys()))
        else:
            if len(numbers) > 1:
                print(
                    Fore.YELLOW
                    + f"Extracted numbers {numbers} is not exactly one. Select the first one."
                    + Style.RESET_ALL
                )
                prediction = numbers[0]
            else:
                print(
                    Fore.RED
                    + f"Prediction {pred_text} has no extracted numbers. Randomly select one."
                    + Style.RESET_ALL
                )
                prediction = random.choice(list(label2desc.keys()))
        return str(prediction)


if __name__ == "__main__":
    from argparse import ArgumentParser

    from execution_pipeline import main

    parser = ArgumentParser()
    parser.add_argument("--bench_name", type=str, required=True)
    parser.add_argument("--model_name1", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    # parser.add_argument("--model_name2", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model_name2", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_8bit", action="store_true")
    parser.add_argument(
        "--output_path", type=str, default=None, help="path to save csv file for kaggle submission"
    )
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    if args.bench_name.startswith("classification"):
        max_tokens = 128
        agent_name = ClassificationAgent
    elif args.bench_name.startswith("sql_generation"):
        max_tokens = 512
        agent_name = SQLGenerationAgent
    else:
        raise ValueError(f"Invalid benchmark name: !{args.bench_name}!")
    # Classification: Medical diagnosis; SQL generation: Text-to-SQL
    bench_cfg = {"bench_name": args.bench_name, "output_path": args.output_path}
    llm_config = {
        "model_name1": args.model_name1,
        "model_name2": args.model_name2,
        "exp_name": f"self_streamicl_{args.bench_name}_{args.model_name1}",
        "bench_name": bench_cfg["bench_name"],
        "max_tokens": max_tokens,
        "do_sample": False,
        "device": args.device,
        "use_8bit": args.use_8bit,
        "rag": {
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "seed": 42,
            "top_k": 5,
            "order": "similar_at_top",
        },
    }
    agent = agent_name(llm_config)
    main(
        agent,
        bench_cfg,
        debug=args.debug,
        use_wandb=args.use_wandb,
        wandb_name=llm_config["exp_name"],
        wandb_config=llm_config,
    )
