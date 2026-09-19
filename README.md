
<div align="center">
  
# Agentic MailBot
**Autonomous Multi-Agent Email
Response System with Contextual Intelligence**


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Status](https://img.shields.io/badge/Status-Active-green.svg)]()

</div>

Agentic MailBot, a novel autonomous email response system that leverages multi-agent architecture, large language models (LLMs), and retrieval-augmented generation (RAG) to provide contextually appropriate responses to incoming business emails without
human intervention. The system employs a sophisticated classification mechanism to categorize
emails into customer service queries (Type A) and order lifecycle inquiries (Type B), subsequently routing them through specialized processing pipelines. Type A emails utilize RAGbased retrieval from vector databases containing company policies and product information,
while Type B emails employ ReAct agents with SQL database toolkit integration for dynamic
data retrieval. Our implementation demonstrates significant improvements in response accuracy, consistency, and processing speed compared to traditional manual email handling systems.
The framework is built using LangChain, LangGraph, Python, and SQLite, providing a scalable
and maintainable solution for enterprise-level email automation.


# System Architecture

Agentic MailBot employs a modular, multi-agent architecture designed to handle the complexity
and variability of business email communications. The system consists of four primary components:
Email Ingestion, Classification Agent, Type-Specific Processing Agents, and Email Dispatch.

## Email Ingestion Module
The email ingestion module serves as the entry point for all incoming communications. It implements standard email protocols (IMAP/POP3) to automatically fetch new messages from designated
mailboxes. The module includes preprocessing capabilities to extract metadata, clean email content,
and prepare messages for classification.



```mermaid
flowchart TD
    A[Start Email Processing System] --> B[Fetch All Unread Emails]
    B --> C{Any Unread Emails?}
    
    C -->|No| D[Wait 10 Minutes]
    D --> B
    
    C -->|Yes| E[Create Email Queue<br/>Store all emails in dictionary]
    E --> F[Initialize Counters<br/>processed_count = 0<br/>failed_count = 0]
    
    F --> G[Get Next Email from Queue]
    G --> H{More Emails<br/>in Queue?}
    
    H -->|No| I[Display Batch Summary<br/>Show processed/failed counts]
    I --> D
    
    H -->|Yes| J[Extract Email Info<br/>- Subject<br/>- Body<br/>- Sender Email<br/>- Sender Name]
    
    J --> K[Classify Email<br/>using LLM Classifier]
    K --> L{Email Type?}
    
    L -->|Type A<br/>Customer Service| M[Type A Writer<br/>- Use FAISS Vector DB<br/>- Retrieve Context<br/>- Generate Response]
    
    L -->|Type B<br/>Order/Payment| N[Type B Writer<br/>- Use SQL Database<br/>- Query Order Data<br/>- Generate Response]
    
    M --> O[Generate Email Response<br/>- Subject<br/>- Body with signature]
    N --> O
    
    O --> P[Send Email Response<br/>to Customer]
    P --> Q{Email Sent<br/>Successfully?}
    
    Q -->|Yes| R[Increment processed_count<br/>Log Success]
    Q -->|No| S[Increment failed_count<br/>Log Error]
    
    R --> T[Wait 2 seconds<br/>Rate limiting]
    S --> T
    T --> G
    
    style A fill:#21262d,stroke:#58a6ff,stroke-width:2px,color:#ffffff
    style B fill:#0d1117,stroke:#7c3aed,stroke-width:2px,color:#ffffff
    style C fill:#161b22,stroke:#f85149,stroke-width:2px,color:#ffffff
    style D fill:#21262d,stroke:#da3633,stroke-width:2px,color:#ffffff
    style E fill:#0d1117,stroke:#238636,stroke-width:2px,color:#ffffff
    style F fill:#161b22,stroke:#d29922,stroke-width:2px,color:#ffffff
    style G fill:#0d1117,stroke:#7c3aed,stroke-width:2px,color:#ffffff
    style H fill:#161b22,stroke:#f85149,stroke-width:2px,color:#ffffff
    style I fill:#21262d,stroke:#58a6ff,stroke-width:2px,color:#ffffff
    style J fill:#0d1117,stroke:#238636,stroke-width:2px,color:#ffffff
    style K fill:#161b22,stroke:#f78166,stroke-width:2px,color:#ffffff
    style L fill:#161b22,stroke:#f85149,stroke-width:2px,color:#ffffff
    style M fill:#0d1117,stroke:#238636,stroke-width:2px,color:#ffffff
    style N fill:#0d1117,stroke:#1f6feb,stroke-width:2px,color:#ffffff
    style O fill:#161b22,stroke:#d29922,stroke-width:2px,color:#ffffff
    style P fill:#0d1117,stroke:#7c3aed,stroke-width:2px,color:#ffffff
    style Q fill:#161b22,stroke:#f85149,stroke-width:2px,color:#ffffff
    style R fill:#0d1117,stroke:#238636,stroke-width:2px,color:#ffffff
    style S fill:#21262d,stroke:#da3633,stroke-width:2px,color:#ffffff
    style T fill:#161b22,stroke:#6e7681,stroke-width:2px,color:#ffffff
    
    classDef typeA fill:#0d1117,stroke:#238636,stroke-width:3px,color:#ffffff
    classDef typeB fill:#0d1117,stroke:#1f6feb,stroke-width:3px,color:#ffffff
    classDef decision fill:#161b22,stroke:#f85149,stroke-width:3px,color:#ffffff
    classDef process fill:#0d1117,stroke:#7c3aed,stroke-width:3px,color:#ffffff
    classDef wait fill:#21262d,stroke:#da3633,stroke-width:3px,color:#ffffff
    
    class M typeA
    class N typeB
    class C,H,L,Q decision
    class B,G,J,K,O,P process
    class D,T wait
```


## Classification Agent

The classification agent represents the system’s decision-making core, responsible for categorizing
incoming emails into two distinct types:
Type A - Customer Service Inquiries:
• Return and refund policy questions
• Product complaints and issues
• Exchange requests
• Product availability and sizing inquiries
• Account-related problems
• General customer support queries
Type B - Order Lifecycle and Delivery:
• Order confirmation and status updates
• Payment-related questions
• Tracking and shipping information
• Delivery time estimates
• Lost or delayed shipment reports
• Billing and invoice inquiries
• Order cancellation or modification requests
• Refund payment confirmations
• Delivery instruction issues
The classifier employs a fine-tuned transformer model trained on a comprehensive dataset of
labeled business emails. The model achieves high accuracy through careful feature engineering and
domain-specific training.


## Type A Processing Agent (RAG-Based)

Type A emails are processed through a sophisticated RAG-based pipeline that combines the reasoning capabilities of large language models with domain-specific knowledge retrieval. The agent
maintains access to a comprehensive vector database containing:
• Company policies and procedures
• Product specifications and documentation
• Terms and conditions
• Frequently asked questions
• Historical resolution patterns

The system architecture employs LangGraph for workflow orchestration, creating a directed
acyclic graph (DAG) that manages the complete email processing pipeline. The state management
system maintains context throughout the processing flow, ensuring data consistency and proper
error handling.
The classifier agent uses structured output parsing with Pydantic models to ensure reliable
categorization. The Type A agent integrates FAISS vector database for semantic similarity search,
while the Type B agent leverages the ReAct framework with SQLDatabaseToolkit for dynamic
database interactions. The system maintains complete email thread context and implements proper
authentication for Gmail integration using IMAP/SMTP protocols

## Type B Processing Agent (ReAct-Based)
Type B emails require access to dynamic, real-time information stored in relational databases. The
Type B agent implements the ReAct (Reasoning and Acting) paradigm, combining logical reasoning
with tool usage capabilities. The agent has access to SQLDatabaseToolkit, which provides the
following tools:
• sql_db_query: Executes SQL queries and returns results
• sql_db_schema: Retrieves table schemas and sample data
• sql_db_list_tables: Lists available database tables
• sql_db_query_checker: Validates SQL queries before execution
The Type B agent implements a comprehensive ReAct loop that includes: 
1. **Analysis Phase**: Understanding email content and determining data requirements
2. **Planning Phase**:Identifying relevant database tables and required queries
3. **Execution Phase**: Using SQLDatabaseToolkit with four key tools:
   - sql_db_list_tables: Discovers available database tables
   - sql_db_schema: Retrieves table schemas and sample data
   - sql_db_query_checker: Validates SQL queries before execution
   - sql_db_query: Executes validated queries and returns results
4.**Integration Phase**: Combining query results with email context
5. **Response Generation**:Creating structured email responses with proper formatting
The agent maintains safety by preventing DML operations and limiting query results to prevent
resource exhaustion. All responses include standardized company branding and contact information.


## we will walk through how an agent that can answer questions about a SQL
database.:
1. Fetch the available tables from the database
2. Decide which tables are relevant to the question
3. Fetch the schemas for the relevant tables
4. Generate a query based on the question and information from the schemas
5. Double-check the query for common mistakes using an LLM
6. Execute the query and return the results
7. Correct mistakes surfaced by the database engine until the query is successful
8. Formulate a response based on the results

## Input Email and AI-Generated Output Email

The following figures demonstrate the practical application of the Agentic MailBot system, showing
both the original customer email and the corresponding AI-generated response.
The first pair of images (Figures 2 and 3) demonstrates the system’s Type B agent capabilities
for handling order-related queries. The customer’s inquiry about payment confirmation for order ID
7 triggered the SQL-based agent, which successfully queried the database to retrieve the payment
status. The agent determined that the payment had failed and provided an appropriate response
with clear next steps for the customer.
The second pair of images (Figures 4 and 5) showcases the Type A agent’s performance in
![input_email](https://github.com/user-attachments/assets/68b3d756-d3c1-4f5e-94b4-f34c9311eb85)
Figure 2: Input email received by the system showing customer inquiry about payment confirmation
for order ID 7.

![output_email](https://github.com/user-attachments/assets/06785fef-9b41-4638-8e11-36238bac94f4)

Figure 3: AI-generated structured output email providing payment status information and follow-up
instructions.

handling policy-related inquiries. Shubham’s question about return options represents a typical
customer service scenario where policy information needs to be retrieved and explained. The RAGbased agent successfully identified this as a return policy query and provided comprehensive guidance
about the return process, including the time limitations and account navigation instructions.
This example demonstrates the system’s ability to handle ambiguous queries where the customer
doesn’t provide specific order details but still expects helpful guidance. The response includes both
general return policy information and a request for more specific details (order number) to provide
further assistance, showcasing the agent’s conversational intelligence and problem-solving approach.
The implementation demonstrates the system’s ability to process natural language queries, execute database operations, and generate contextually appropriate responses. The first example shows
how the agent successfully identified the payment failure status for order ID 7 and provided a comprehensive response with appropriate follow-up actions. The second example illustrates the system’s
capability to handle return policy inquiries by providing clear guidance on the return process and
policy limitations.



![input_email2](https://github.com/user-attachments/assets/7a6a772c-e960-486b-b81f-f0fb09c3cf95)

Figure 4: Input email from customer Shubham inquiring about return options for an order.


![output_image2](https://github.com/user-attachments/assets/af2979c1-a232-4b09-8910-5bad5812c1c2)

Figure 5: AI-generated response providing return policy information and guidance for order returns
