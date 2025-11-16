"""
Tool Executor

This module handles the execution of tools when called by the AI agent.
It maps tool names to their corresponding functions and executes them with the provided parameters.
"""

from typing import Dict, Any, Callable
from .image_tools import image_generator, image_uploader
from .blog_tools import blog_creator, blog_inserter, calculate_read_time, generate_schema_markup


class ToolExecutor:
    """
    Executes tools based on their names and parameters

    This class provides a clean interface for the agent to execute tools.
    It handles the mapping between tool names and their implementations.
    """

    def __init__(self):
        """Initialize the tool executor with available tools"""
        self.tools = {
            "image_generator": image_generator,
            "image_uploader": image_uploader,
            "blog_creator": blog_creator,
            "blog_inserter": blog_inserter
        }

        # Helper functions (not exposed as tools to AI, but available for internal use)
        self.helpers = {
            "calculate_read_time": calculate_read_time,
            "generate_schema_markup": generate_schema_markup
        }

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with the given input

        Args:
            tool_name: Name of the tool to execute
            tool_input: Dictionary of input parameters for the tool

        Returns:
            Dictionary with the tool execution results
        """
        if tool_name not in self.tools:
            return {
                "status": "error",
                "message": f"Unknown tool: {tool_name}"
            }

        try:
            # Get the tool function
            tool_function = self.tools[tool_name]

            # Execute the tool with the provided input
            result = tool_function(**tool_input)

            return result

        except TypeError as e:
            return {
                "status": "error",
                "message": f"Invalid parameters for tool {tool_name}: {str(e)}"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error executing tool {tool_name}: {str(e)}"
            }

    def get_available_tools(self) -> list:
        """Get list of available tool names"""
        return list(self.tools.keys())