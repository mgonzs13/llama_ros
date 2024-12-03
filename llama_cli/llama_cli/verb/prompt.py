# MIT License
#
# Copyright (c) 2024  Miguel Ángel González Santamarta
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from ros2cli.verb import VerbExtension
from llama_cli.api import prompt_llm, positive_float


class PromptVerb(VerbExtension):

    def add_arguments(self, parser, cli_name):
        arg = parser.add_argument("prompt", help="prompt text for the LLM")
        parser.add_argument(
            "-r",
            "--reset",
            action="store_true",
            help="Whether to reset the LLM and its context before prompting",
        )
        parser.add_argument(
            "-t",
            "--temp",
            metavar="N",
            type=positive_float,
            default=0.8,
            help="Temperature value (default: 0.8)",
        )
        parser.add_argument(
            "--image-url", type=str, default="", help="Image URL to sent to the VLM"
        )

    def main(self, *, args):
        prompt_llm(
            args.prompt, reset=args.reset, temp=args.temp, image_url=args.image_url
        )
