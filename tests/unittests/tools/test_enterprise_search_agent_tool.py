# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from unittest import mock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.plugins.plugin_manager import PluginManager
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.enterprise_search_agent_tool import create_enterprise_search_agent
from google.adk.tools.enterprise_search_agent_tool import EnterpriseSearchAgentTool
from google.adk.tools.tool_context import ToolContext
from pytest import mark


async def _create_tool_context() -> ToolContext:
  session_service = InMemorySessionService()
  session = await session_service.create_session(
      app_name='test_app', user_id='test_user'
  )
  agent = SequentialAgent(name='test_agent')
  invocation_context = InvocationContext(
      invocation_id='invocation_id',
      agent=agent,
      session=session,
      session_service=session_service,
      artifact_service=InMemoryArtifactService(),
      memory_service=InMemoryMemoryService(),
      plugin_manager=PluginManager(),
      run_config=RunConfig(),
  )
  return ToolContext(invocation_context=invocation_context)


class TestEnterpriseSearchAgentTool:
  """Test the EnterpriseSearchAgentTool class."""

  def test_create_enterprise_search_agent(self):
    """Test that create_enterprise_search_agent creates a valid agent."""
    agent = create_enterprise_search_agent('gemini-pro')
    assert isinstance(agent, LlmAgent)
    assert agent.name == 'enterprise_search_agent'
    assert 'enterprise_web_search' in [t.name for t in agent.tools]

  def test_enterprise_search_agent_tool_init(self):
    """Test initialization of EnterpriseSearchAgentTool."""
    mock_agent = mock.MagicMock(spec=LlmAgent)
    mock_agent.name = 'test_agent'
    mock_agent.description = 'test_description'
    tool = EnterpriseSearchAgentTool(mock_agent)
    assert tool.agent == mock_agent

  @mark.asyncio
  @mock.patch('google.adk.tools._search_agent_tool._SearchAgentTool.run_async')
  async def test_run_async_succeeds(self, mock_run_async):
    """Test that run_async calls the base class method."""
    # Arrange
    mock_agent = mock.MagicMock(spec=LlmAgent)
    mock_agent.name = 'enterprise_search_agent'
    mock_agent.description = 'test_description'
    mock_agent.input_schema = None
    mock_agent.output_schema = None

    tool = EnterpriseSearchAgentTool(mock_agent)
    tool_context = await _create_tool_context()
    mock_run_async.return_value = 'test response'

    # Act
    result = await tool.run_async(
        args={'request': 'test query'}, tool_context=tool_context
    )

    # Assert
    mock_run_async.assert_called_once_with(
        args={'request': 'test query'}, tool_context=tool_context
    )
    assert result == 'test response'
