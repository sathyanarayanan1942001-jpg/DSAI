import asyncio
import sys

try:
    from google.antigravity import Agent, LocalAgentConfig, CapabilitiesConfig
except ImportError:
    print(
        "Error: 'google-antigravity' SDK is not installed in the current Python environment.\n"
        "To install, please run:\n"
        "    pip install google-antigravity\n"
    )
    sys.exit(1)


async def run_blinkit_diagnosis_agent():
    # 1. Configure the Antigravity local agent for Anna Nagar dark store optimization
    config = LocalAgentConfig(
        system_instructions=(
            "You are an expert operations and logistics AI agent specialized in quick-commerce "
            "dark store optimization. Analyze order bottlenecks, packing delays, and delivery "
            "routing constraints for the Blinkit dark store in Anna Nagar, Chennai."
        ),
        capabilities=CapabilitiesConfig()
    )

    # 2. Spawning the agent using the asynchronous context manager
    async with Agent(config) as agent:
        prompt = (
            "Perform a root cause diagnosis for delivery delays at the Blinkit Anna Nagar "
            "Chennai hub. Focus on peak-hour traffic bottlenecks around 2nd Avenue and Shanthi Colony, "
            "picker congestion inside the dark store, and recommend an automated dispatch algorithm."
        )

        print("=" * 70)
        print("  Initializing Antigravity Agent: Blinkit Anna Nagar Operations")
        print("=" * 70 + "\n")

        # Chat returns an asynchronous streaming response object
        response = await agent.chat(prompt)

        # 3. Stream reasoning / internal diagnostic thoughts in real-time
        if hasattr(response, "thoughts"):
            print("\033[94m[Agent Reasoning & Diagnostic Logic]\033[0m")
            async for thought in response.thoughts:
                sys.stdout.write(f"\033[90m{thought}\033[0m")
                sys.stdout.flush()
            print("\n" + "-" * 70 + "\n")

        # 4. Stream final response tokens
        print("\033[92m[Operational Optimization Report]\033[0m")
        async for token in response:
            sys.stdout.write(token)
            sys.stdout.flush()
        print("\n")


if __name__ == "__main__":
    asyncio.run(run_blinkit_diagnosis_agent())
