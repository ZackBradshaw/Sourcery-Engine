import time
import types
from typing import Any, Dict, List, Tuple, Union
from langchain.agents import AgentExecutor
from langchain.input import get_color_mapping
from langchain.schema import AgentAction, AgentFinish
from .translator import Translator


class AgentExecutorWithTranslation(AgentExecutor):
    translator: Translator = Translator()

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        try:
            outputs = super().prep_outputs(inputs, outputs, return_only_outputs)
        except ValueError as e:
            return outputs
        else:
            if "input" in outputs:
                outputs = self.translator(outputs)
            return outputs


class Executor(AgentExecutorWithTranslation):
    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green"]
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Let's start tracking the iterations the agent has gone through
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        # We now enter the agent loop (until it returns something).
        while self._should_continue(iterations, time_elapsed):
            next_step_output = self._take_next_step(
                name_to_tool_map, color_mapping, inputs, intermediate_steps
            )
            if isinstance(next_step_output, AgentFinish):
                yield self._return(next_step_output, intermediate_steps)
                return

            for i, output in enumerate(next_step_output):
                agent_action = output[0]
                tool_logo = None
                for tool in self.tools:
                    if tool.name == agent_action.tool:
                        tool_logo = tool.tool_logo_md
                if isinstance(output[1], types.GeneratorType):
                    logo = f"{tool_logo}" if tool_logo is not None else ""
                    yield (
                        AgentAction("", agent_action.tool_input, agent_action.log),
                        f"Further use other tool {logo} to answer the question.",
                    )
                    for out in output[1]:
                        yield out
                    next_step_output[i] = (agent_action, out)
                else:
                    for tool in self.tools:
                        if tool.name == agent_action.tool:
                            yield (
                                AgentAction(
                                    tool_logo, agent_action.tool_input, agent_action.log
                                ),
                                output[1],
                            )

            intermediate_steps.extend(next_step_output)
            if len(next_step_output) == 1:
                next_step_action = next_step_output[0]
                # See if tool should return directly
                tool_return = self._get_tool_return(next_step_action)
                if tool_return is not None:
                    yield self._return(tool_return, intermediate_steps)
                    return
            iterations += 1
            time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )
        yield self._return(output, intermediate_steps)
        return

    def __call__(
        self, inputs: Union[Dict[str, Any], Any], return_only_outputs: bool = False
    ) -> Dict[str, Any]:
        """Run the logic of this chain and add to output if desired.

        Args:
            inputs: Dictionary of inputs, or single input if chain expects
                only one param.
            return_only_outputs: boolean for whether to return only outputs in the
                response. If True, only new keys generated by this chain will be
                returned. If False, both input keys and new keys generated by this
                chain will be returned. Defaults to False.

        """
        inputs = self.prep_inputs(inputs)

        try:
            for output in self._call(inputs):
                if type(output) is dict:
                    output = self.prep_outputs(inputs, output, return_only_outputs)
                yield output
        except (KeyboardInterrupt, Exception) as e:
            raise e
        return self.prep_outputs(inputs, output, return_only_outputs)
        return output
