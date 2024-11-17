# Copyright 2024 Iguazio
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
from abc import ABC, abstractmethod
import subprocess
import requests
from typing import Any, Dict, List, Union

import mlrun
from mlrun.common.schemas.api_gateway import APIGatewayAuthenticationMode
from mlrun.utils import logger


class Application(ABC):
    def __init__(self, name: str, project_name: str = None, *args, **kwargs):
        self._name = name
        self._project = (
            mlrun.get_or_create_project(project_name)
            if project_name
            else mlrun.get_current_project()
        )
        self._application_runtime = None

    @abstractmethod
    def deploy(self, *args, **kwargs):
        pass

    @abstractmethod
    def invoke(self, *args, **kwargs) -> dict:
        pass

    def is_deployed(self):
        return self._application_runtime is not None


class NIMApplication(Application):
    def __init__(
        self,
        name: str,
        model_name: str,
        project_name: str = None,
        image_name: str = None,
        ngc_api_key: str = None,
        generation_configuration: dict = None,
    ):
        super().__init__(name=name, project_name=project_name)
        self._model_name = model_name
        self._image_name = image_name or f"nvcr.io/nim/{model_name}:latest"
        self._ngc_api_key = ngc_api_key
        self._generation_configuration = generation_configuration or {}

        self._docker_creds_secret_name = None
        self._ngc_secret_name = None

        try:
            self._application_runtime = self._project.get_function(
                key=self._name, ignore_cache=True
            )
            logger.info(
                f"Found an existing application. Status: {self._application_runtime.status.to_json()}"
            )
        except requests.exceptions.HTTPError:
            if not ngc_api_key:
                raise ValueError(
                    "NGC API key is required to deploy the NIM application."
                )

    def deploy(
        self,
        force_redeploy: bool = False,
        application_internal_application_port: int = 8000,
        application_node_selection: dict = None,
        api_gateway_path: str = None,
        api_gateway_direct_port_access: bool = False,
        api_gateway_authentication_mode: APIGatewayAuthenticationMode = None,
        api_gateway_authentication_creds: tuple[str, str] = None,
        api_gateway_ssl_redirect: bool = None,
        api_gateway_set_as_default: bool = False,
    ):
        if not force_redeploy and self.is_deployed():
            return
        self._set_secrets()
        self._deploy_application(
            internal_application_port=application_internal_application_port,
            node_selection=application_node_selection,
        )
        self._create_api_gateway(
            path=api_gateway_path,
            direct_port_access=api_gateway_direct_port_access,
            authentication_mode=api_gateway_authentication_mode,
            authentication_creds=api_gateway_authentication_creds,
            ssl_redirect=api_gateway_ssl_redirect,
            set_as_default=api_gateway_set_as_default,
        )  # TODO: set_as_default=True to skip syncing

    def _deploy_application(
        self,
        internal_application_port: int = 8000,
        node_selection: dict = None,
    ):
        if not self._ngc_secret_name or not self._docker_creds_secret_name:
            raise Exception("Secrets are not created. Can not deploy the application.")

        application_runtime = self._project.set_function(
            name=self._name, kind="application", image=self._image_name
        )
        application_runtime.set_internal_application_port(
            port=internal_application_port
        )
        application_runtime.set_env_from_secret(
            secret=self._ngc_secret_name, name="NGC_API_KEY"
        )
        application_runtime.spec.env.append(
            {
                "name": "LD_LIBRARY_PATH",
                "value": "/usr/local/lib/python3.10/dist-packages/tensorrt_llm/libs:"
                "/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:"
                "/usr/local/lib/python3.10/dist-packages/tensorrt_libs",
            }
        )
        application_runtime.set_image_pull_configuration(
            image_pull_secret_name=self._docker_creds_secret_name
        )
        if node_selection:
            application_runtime.with_node_selection(node_selector=node_selection)
        application_runtime.deploy(create_default_api_gateway=False)
        self._application_runtime = application_runtime

    def _create_api_gateway(
        self,
        name: str = None,
        path: str = None,
        direct_port_access: bool = False,
        authentication_mode: APIGatewayAuthenticationMode = None,
        authentication_creds: tuple[str, str] = None,
        ssl_redirect: bool = None,
        set_as_default: bool = False,
    ):
        """
        Create the application API gateway. Once the application is deployed, the API gateway can be created.
        An application without an API gateway is not accessible.

        :param name:                    The name of the API gateway, defaults to <function-name>-<function-tag>
        :param path:                    Optional path of the API gateway, default value is "/"
        :param direct_port_access:      Set True to allow direct port access to the application sidecar
        :param authentication_mode:     API Gateway authentication mode
        :param authentication_creds:    API Gateway basic authentication credentials as a tuple (username, password)
        :param ssl_redirect:            Set True to force SSL redirect, False to disable. Defaults to
                                        mlrun.mlconf.force_api_gateway_ssl_redirect()
        :param set_as_default:          Set the API gateway as the default for the application (`status.api_gateway`)

        :return:    The API gateway URL
        """
        if not self.is_deployed():
            raise "API gateway can not be created - Application is not deployed."

        name = name or f"{self._name}-gw"

        self._application_runtime.create_api_gateway(
            name=name,
            path=path,
            direct_port_access=direct_port_access,
            authentication_mode=authentication_mode,
            authentication_creds=authentication_creds,
            ssl_redirect=ssl_redirect,
            set_as_default=set_as_default,
        )

        self._application_runtime._sync_api_gateway()
        api_gateway = self._project.get_api_gateway(name=name)
        self._application_runtime.api_gateway = api_gateway

    def _set_secrets(self):
        self._create_docker_creds_secret()
        self._create_secret_with_api_key()

    def _create_docker_creds_secret(self):
        # Command which creates a secret to pull NIM image
        self._docker_creds_secret_name = (
            f"{self._project.name}-{self._name}-nim-creds".replace("/", "-")
        )
        self._execute_command(
            command=[
                "kubectl", 
                "delete", 
                "secret", 
                self._docker_creds_secret_name
            ], 
            ignore_error=True
        )
        self._execute_command(
            command=[
                "kubectl",
                "create",
                "secret",
                "docker-registry",
                self._docker_creds_secret_name,
                "--docker-server=nvcr.io",
                r"--docker-username=\$oauthtoken",
                f"--docker-password={self._ngc_api_key}",
                "--namespace=default-tenant",
            ], 
            ignore_error=False
        )

    def _create_secret_with_api_key(self):
        self._ngc_secret_name = (
            f"{self._project.name}-{self._name}-ngc-api-key".replace("/", "-")
        )
        self._execute_command(
            command=[
                "kubectl", 
                "delete", 
                "secret", 
                self._ngc_secret_name
            ], 
            ignore_error=True
        )
        self._execute_command(
            command=[
                "kubectl",
                "create",
                "secret",
                "generic",
                self._ngc_secret_name,
                f"--from-literal=NGC_API_KEY={self._ngc_api_key}",
                "--namespace=default-tenant",
            ], 
            ignore_error=False
        )

    @staticmethod
    def _execute_command(command: list[str], ignore_error: bool):
        command = " ".join(command)
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            error = f"Failed to execute command '{command}': {result.stderr}"
            logger.error(error)
            if not ignore_error:
                raise Exception(error)

    def invoke(
        self,
        messages: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        path="/v1/chat/completions",
        model_name: str = None,
        **generation_configuration,
    ):
        # Normalize messages to a list of dictionaries
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            raise TypeError("Messages should be a string, dict, or list of dicts")

        if not self.is_deployed():
            raise Exception("Application isn't deployed")
        body = {
            "model": model_name or self._model_name,
            "messages": messages,
            **self._generation_configuration,
            **generation_configuration,
        }
        return self._application_runtime.invoke(path=path, body=body, method="POST")
    
    def get_url(self) -> str:
        return self._application_runtime.status.external_invocation_urls[0]
