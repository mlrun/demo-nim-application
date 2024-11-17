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
import mlrun


def setup(
    project: mlrun.projects.MlrunProject,
) -> mlrun.projects.MlrunProject:
    """
    Creating the project for the demo. This function is expected to call automatically when calling the function
    `mlrun.get_or_create_project`.

    :param project: The project to set up.

    :returns: A fully prepared project for this demo.
    """

    # Adding secrets to the projects:
    project.set_secrets(
        {
            "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
            "OPENAI_API_BASE": os.environ["OPENAI_BASE_URL"],
        }
    )

    # Set functions
    _set_function(
        project=project,
        func="nim_model_server.py",
        name="nim-gateway",
        kind="serving",
        image="mlrun/mlrun",
    )

    # Save and return the project:
    project.save()
    return project


def _set_function(
    project: mlrun.projects.MlrunProject,
    func: str,
    name: str,
    kind: str,
    image: str = None,
):
    # Set the given function:
    mlrun_function = project.set_function(
        func=func,
        name=name,
        kind=kind,
        with_repo=False,
        image=image,
    ).apply(mlrun.auto_mount())

    mlrun_function.save()