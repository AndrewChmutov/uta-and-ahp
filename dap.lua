local dap = require("dap")

dap.configurations.python = {
    {
        type = "python",
        request = "launch",
        name = "UTA",
        module = "uta_ahp",
        console = "integratedTerminal",
        args = { "uta" },
    },
}
