# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from genai_factory.chains.base import ChainRunner
from genai_factory.chains.retrieval import MultiRetriever


class GeneralAgent(ChainRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._llm = None
        self.agent = None
        self.retriever = None

    @property
    def llm(self):
        if not self._llm:
            self._llm = ChatOpenAI(model="gpt-4", temperature=0.5)
        return self._llm

    def _get_agent(self):
        if self.agent:
            return self.agent
        # Create the RAG tools
        retriever = MultiRetriever(default_collection="default", context=self.context)
        retriever.post_init()
        self.retriever = retriever._get_retriever("default")
        data_retriever_tool = create_retriever_tool(
            self.retriever.chain.retriever.vectorstore.as_retriever(),
            "bank-policy-data-retriever",
            "Query a retriever to get information about the bank's policies.",
        )
        tools = [data_retriever_tool]
        llm_with_tools = self.llm.bind_tools(tools)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    TOOL_PROMPT,
                ),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
        return AgentExecutor(
            agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
        )

    def _run(self, event):
        self.agent = self._get_agent()
        response = list(self.agent.stream({"input": event.query}))
        answer = response[-1]["messages"][-1].content
        return {"answer": answer, "sources": ""}


TOOL_PROMPT = str(
    "You are a helpful AI banking assistant that can provide information about the banks policies and answer general questions. "
    "Try to answer the question using the data retriever, if the query is related to bank policy, "
    "if you can't find the answer, make something up."
    "Do not answer any questions that are not related to the bank, or that are inappropriate."
)
