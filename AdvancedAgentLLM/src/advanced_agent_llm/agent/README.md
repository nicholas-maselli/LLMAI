# Agent layer (not implemented yet)

This package will eventually contain two distinct ideas:

1. A structured representation of tools and tool-call messages.
2. A runtime loop that lets the model request a tool, executes it safely, and
   returns the result to the model as another message.

The transformer does not execute tools. SFT can teach it to emit a structured
tool request, while this runtime performs the real action.

