# FlexUSFL

FlexUSFL is a research framework for U-shaped federated split learning with large language models.

The framework splits an LLM into three parts: a client-side head model, a server-side middle model, and a client-side tail model. Raw data remains on clients, while only intermediate activations and gradients are exchanged with the server.
