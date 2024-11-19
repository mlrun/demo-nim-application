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
import os
import sys

# Add the directory containing your modules to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from genai_factory.chains.base import HistorySaver, SessionLoader
from genai_factory.chains.refine import RefineQuery
from genai_factory import workflow_server

from classifier_llm import Classifier
from loans_agent import LoanAgent
from investment_agent import InvestmentAgent
from general_agent import GeneralAgent
from choice_intent import MyChoice

# We have 3 intent categories
intent_categories = ["loans", "investments", "other"]

# This is the workflow graph of our application
workflow_graph = [
    SessionLoader(),
    RefineQuery(),
    Classifier(classifier_classes=intent_categories),
    MyChoice(),
    (
        GeneralAgent(),
        LoanAgent(),
        InvestmentAgent(),
    ),
    HistorySaver(),
]

# Add our workflow to the server to be deployed
workflow_server.add_workflow(
    name="default",
    graph=workflow_graph,
    workflow_type="application",
)