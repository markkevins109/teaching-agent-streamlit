import streamlit as st
import json
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq LLM Provider
class LLMProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "meta-llama/llama-4-maverick-17b-128e-instruct"
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.conversation_history = []
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate_response(self, system_prompt: str, user_message: str) -> str:
        """Generate response using Groq API"""
        try:
            # Build messages for the API call
            messages = [
                {"role": "system", "content": system_prompt}
            ]

            # Add conversation history
            for msg in self.conversation_history[-10:]:  # Keep last 10 messages for context
                messages.append(msg)

            # Add current user message if provided
            if user_message.strip():
                messages.append({"role": "user", "content": user_message})

            # API request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500,
                "top_p": 0.9,
                "stream": False
            }

            # Make API call
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]

                # Update conversation history
                if user_message.strip():
                    self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})

                return assistant_message
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return self._fallback_response(system_prompt, user_message)

        except Exception as e:
            st.error(f"Error calling Groq API: {e}")
            return self._fallback_response(system_prompt, user_message)

    def _fallback_response(self, system_prompt: str, user_message: str) -> str:
        """Fallback responses when API fails"""
        responses = {
            "greeting": "Hello! Today we're going to explore oscillatory motion. Think of a child's swing - it goes forward and then back to the same spot. What would you call that kind of motion?",
            "definition": "Good start! In science we call one complete forward-and-back trip an oscillation. An oscillatory motion is one that repeats in equal time intervals—like a steady heartbeat. Does that make sense?",
            "exploration": "Exactly! Now, why doesn't the swing just stop in the middle? What pulls it back?",
            "misconception": "That's a common idea! Let's test it: imagine two identical swings, one kid heavier than the other but both pushed the same. They actually take almost the same time to go back and forth. So weight isn't the big factor here.",
            "context": "Because each oscillation is so regular, clock-makers once used pendulums to mark seconds. Pretty clever, huh?",
            "transfer": "Critical thinking: If we took the swing to a place with no gravity, would it still oscillate?",
            "quiz": "Quick recap: oscillation = repeating motion; gravity restores; inertia carries through. Ready for a couple of quiz questions? 1) True/False: A heavier pendulum swings faster. 2) What force pulls the pendulum back to center?"
        }

        # Simple keyword matching for fallback
        if "greeting" in system_prompt.lower() or "start" in system_prompt.lower():
            return responses["greeting"]
        elif "definition" in system_prompt.lower() or "concept" in system_prompt.lower():
            return responses["definition"]
        elif "exploration" in system_prompt.lower() or "why" in system_prompt.lower():
            return responses["exploration"]
        elif "misconception" in system_prompt.lower():
            return responses["misconception"]
        elif "context" in system_prompt.lower() or "real" in system_prompt.lower():
            return responses["context"]
        elif "transfer" in system_prompt.lower() or "critical" in system_prompt.lower():
            return responses["transfer"]
        elif "quiz" in system_prompt.lower() or "retrieval" in system_prompt.lower():
            return responses["quiz"]
        else:
            return "I understand. Can you tell me more about that?"

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

class PedagogicalState(Enum):
    START = "START"
    APK = "APK"  # Activate Prior Knowledge
    CI = "CI"    # Concept Introduction
    GE = "GE"    # Guided Exploration
    MH = "MH"    # Misconception Handling
    AR = "AR"    # Application & Retrieval
    TC = "TC"    # Transfer & Critical Thinking
    RLC = "RLC"  # Real-Life Context
    RCT = "RCT"  # Reinforcement/Creative Task
    END = "END"

@dataclass
class ConceptPackage:
    """Contains all the content for teaching a specific concept"""
    concept_name: str
    hook_question: str
    one_line_definition: str
    mechanism_question: str
    common_misconception: str
    misconception_correction: str
    real_life_fact: str
    transfer_question: str
    creative_task: str
    misconceptions_regex: List[str] = field(default_factory=list)

@dataclass
class LearnerState:
    """Tracks the current state of the learner"""
    user_msg: str = ""
    current_state: PedagogicalState = PedagogicalState.START
    prior_knowledge: str = ""
    definition_echoed: bool = False
    misconception_detected: bool = False
    retrieval_score: float = 0.0
    transfer_success: bool = False
    context_shared: bool = False
    session_summary: Dict[str, Any] = field(default_factory=dict)

class MemoryStore:
    """Persistent memory for the teaching session"""
    def __init__(self):
        self.memory = {}

    def update(self, updates: Dict[str, Any]):
        self.memory.update(updates)

    def get(self, key: str, default=None):
        return self.memory.get(key, default)

    def clear(self):
        self.memory.clear()

class PedagogicalNode:
    """Generic node that handles different pedagogical moves"""

    def __init__(self, llm_provider: LLMProvider, memory_store: MemoryStore):
        self.llm = llm_provider
        self.memory = memory_store
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        """Load system prompts for each pedagogical state"""
        return {
            PedagogicalState.START: """
You are a friendly tutor starting a lesson. Greet the learner briefly, state today's concept in one sentence, and invite them to begin. Keep it friendly with no jargon.
""",
            PedagogicalState.APK: """
You are activating prior knowledge. Pose one open question that links the concept to the learner's everyday experience. Do not reveal any definition or answer. The question must be answerable with common sense.
""",
            PedagogicalState.CI: """
You are introducing a concept. Provide a concise definition (≤30 words) and ask the learner to restate it. Follow with a rephrase prompt and mention why it matters in one phrase.
""",
            PedagogicalState.GE: """
You are guiding exploration. Ask a "why" or "how" question that makes the learner reason out the mechanism. Do not lecture. After their attempt, give only one nudge or clue if they struggle.
""",
            PedagogicalState.MH: """
You are handling misconceptions. Start on a positive note before correcting. Keep correction ≤2 sentences. Use examples or contrasts to gently fix wrong ideas.
""",
            PedagogicalState.AR: """
You are testing application and retrieval. Ask 2 quick recall questions (T/F, MCQ, or short answer) covering definition and mechanism. Give immediate feedback with ✅/❌ symbols.
""",
            PedagogicalState.TC: """
You are testing transfer and critical thinking. Pose one hypothetical scenario that changes a key variable. Ask learner to predict outcome and justify. Keep scenario ≤2 sentences.
""",
            PedagogicalState.RLC: """
You are providing real-life context. Share one historical, cultural, or modern application (≤3 sentences) that shows relevance. Ask if they've seen it or can think of another use.
""",
            PedagogicalState.END: """
You are concluding the lesson. Summarize 2-3 bullet takeaways. Offer the learner a choice (another concept, review, or end). Use bullet format with no new content.
"""
        }

    def execute(self, state: LearnerState, concept_pkg: ConceptPackage) -> Dict[str, Any]:
        """Execute the pedagogical node and return response with memory updates"""

        system_prompt = self._build_system_prompt(state, concept_pkg)

        # Generate LLM response
        agent_output = self.llm.generate_response(system_prompt, state.user_msg)

        # Process response and update memory
        memory_updates, next_state = self._process_response(state, concept_pkg, agent_output)

        return {
            "agent_output": agent_output,
            "memory_updates": memory_updates,
            "next_state": next_state
        }

    def _build_system_prompt(self, state: LearnerState, concept_pkg: ConceptPackage) -> str:
        """Build system prompt based on current state and concept package"""
        base_prompt = self.prompts[state.current_state]

        # Add concept-specific information
        context = f"""
Concept: {concept_pkg.concept_name}
Current State: {state.current_state.value}
"""

        if state.current_state == PedagogicalState.APK:
            context += f"Hook Question: {concept_pkg.hook_question}\n"
        elif state.current_state == PedagogicalState.CI:
            context += f"Definition: {concept_pkg.one_line_definition}\n"
        elif state.current_state == PedagogicalState.GE:
            context += f"Mechanism Question: {concept_pkg.mechanism_question}\n"
        elif state.current_state == PedagogicalState.MH:
            context += f"Misconception: {concept_pkg.common_misconception}\n"
            context += f"Correction: {concept_pkg.misconception_correction}\n"
        elif state.current_state == PedagogicalState.RLC:
            context += f"Real-life Context: {concept_pkg.real_life_fact}\n"
        elif state.current_state == PedagogicalState.TC:
            context += f"Transfer Question: {concept_pkg.transfer_question}\n"

        return base_prompt + "\n" + context

    def _process_response(self, state: LearnerState, concept_pkg: ConceptPackage, agent_output: str) -> Tuple[Dict[str, Any], PedagogicalState]:
        """Process the response and determine next state"""
        memory_updates = {}
        next_state = state.current_state

        if state.current_state == PedagogicalState.START:
            memory_updates["current_state"] = PedagogicalState.APK
            next_state = PedagogicalState.APK

        elif state.current_state == PedagogicalState.APK:
            memory_updates["prior_knowledge"] = state.user_msg
            next_state = PedagogicalState.CI

        elif state.current_state == PedagogicalState.CI:
            # Check if learner echoed definition correctly
            definition_echoed = self._check_definition_echo(state.user_msg, concept_pkg.one_line_definition)
            memory_updates["definition_echoed"] = definition_echoed
            next_state = PedagogicalState.GE

        elif state.current_state == PedagogicalState.GE:
            # Check for misconceptions
            misconception_detected = self._detect_misconceptions(state.user_msg, concept_pkg.misconceptions_regex)
            memory_updates["misconception_detected"] = misconception_detected
            next_state = PedagogicalState.MH if misconception_detected else PedagogicalState.AR

        elif state.current_state == PedagogicalState.MH:
            memory_updates["misconception_detected"] = False
            next_state = PedagogicalState.AR

        elif state.current_state == PedagogicalState.AR:
            # Score retrieval performance
            retrieval_score = self._score_retrieval(state.user_msg)
            memory_updates["retrieval_score"] = retrieval_score
            next_state = PedagogicalState.TC if retrieval_score >= 0.7 else PedagogicalState.GE

        elif state.current_state == PedagogicalState.TC:
            transfer_success = self._evaluate_transfer(state.user_msg)
            memory_updates["transfer_success"] = transfer_success
            next_state = PedagogicalState.RLC

        elif state.current_state == PedagogicalState.RLC:
            memory_updates["context_shared"] = True
            next_state = PedagogicalState.END

        elif state.current_state == PedagogicalState.END:
            memory_updates["session_summary"] = self._create_session_summary(state)
            next_state = PedagogicalState.END

        return memory_updates, next_state

    def _check_definition_echo(self, user_msg: str, definition: str) -> bool:
        """Check if learner successfully echoed the definition"""
        # Simple keyword matching - in production, use more sophisticated NLP
        key_words = definition.lower().split()
        user_words = user_msg.lower().split()
        overlap = len(set(key_words) & set(user_words))
        return overlap >= len(key_words) * 0.5

    def _detect_misconceptions(self, user_msg: str, misconception_patterns: List[str]) -> bool:
        """Detect common misconceptions using regex patterns"""
        for pattern in misconception_patterns:
            if re.search(pattern, user_msg, re.IGNORECASE):
                return True
        return False

    def _score_retrieval(self, user_msg: str) -> float:
        """Score the learner's retrieval performance"""
        # Simple scoring based on message length and key terms
        # In production, use more sophisticated evaluation
        if len(user_msg.split()) >= 3:
            return 0.8
        return 0.4

    def _evaluate_transfer(self, user_msg: str) -> bool:
        """Evaluate transfer and critical thinking"""
        # Simple evaluation - in production, use LLM-based grading
        return len(user_msg.split()) >= 5

    def _create_session_summary(self, state: LearnerState) -> Dict[str, Any]:
        """Create a summary of the learning session"""
        return {
            "completed_states": [state.current_state.value],
            "retrieval_score": state.retrieval_score,
            "transfer_success": state.transfer_success,
            "misconceptions_handled": state.misconception_detected
        }

class StageRouter:
    """Routes between different pedagogical states"""

    def __init__(self):
        self.transition_graph = {
            PedagogicalState.START: [PedagogicalState.APK],
            PedagogicalState.APK: [PedagogicalState.CI],
            PedagogicalState.CI: [PedagogicalState.GE],
            PedagogicalState.GE: [PedagogicalState.MH, PedagogicalState.AR],
            PedagogicalState.MH: [PedagogicalState.AR],
            PedagogicalState.AR: [PedagogicalState.TC, PedagogicalState.GE],
            PedagogicalState.TC: [PedagogicalState.RLC],
            PedagogicalState.RLC: [PedagogicalState.END],
            PedagogicalState.END: [PedagogicalState.END]
        }

    def get_next_state(self, current_state: PedagogicalState, memory: MemoryStore) -> PedagogicalState:
        """Determine the next pedagogical state based on current state and memory"""
        possible_states = self.transition_graph.get(current_state, [current_state])

        # Decision logic based on memory
        if current_state == PedagogicalState.GE:
            if memory.get("misconception_detected", False):
                return PedagogicalState.MH
            else:
                return PedagogicalState.AR

        elif current_state == PedagogicalState.AR:
            if memory.get("retrieval_score", 0) < 0.7:
                return PedagogicalState.GE
            else:
                return PedagogicalState.TC

        # Default to first possible state
        return possible_states[0] if possible_states else current_state

class TurnTakingController:
    """Controls the conversation flow and manages state transitions"""

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider
        self.memory = MemoryStore()
        self.router = StageRouter()
        self.node = PedagogicalNode(llm_provider, self.memory)
        self.state = LearnerState()

    def process_message(self, user_message: str, concept_pkg: ConceptPackage) -> str:
        """Process a user message and return the agent's response"""

        # Update learner state with user message
        self.state.user_msg = user_message

        # Execute current pedagogical node
        result = self.node.execute(self.state, concept_pkg)

        # Update memory with results
        self.memory.update(result["memory_updates"])

        # Update state
        self.state.current_state = result["next_state"]

        # Update learner state with memory
        for key, value in result["memory_updates"].items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

        return result["agent_output"]

    def reset_session(self):
        """Reset the teaching session"""
        self.memory.clear()
        self.state = LearnerState()

class AdaptiveTeachingAgent:
    """Main teaching agent that orchestrates the entire system"""

    def __init__(self, api_key: str):
        self.llm_provider = LLMProvider(api_key)
        self.controller = TurnTakingController(self.llm_provider)
        self.concept_packages = self._load_concept_packages()

    def _load_concept_packages(self) -> Dict[str, ConceptPackage]:
        """Load concept packages for different topics"""
        return {
            "oscillatory_motion": ConceptPackage(
                concept_name="Oscillatory Motion",
                hook_question="Think of a child's swing. It goes forward and then back to the same spot. What would you call that kind of motion?",
                one_line_definition="An oscillatory motion is one that repeats in equal time intervals—like a steady heartbeat.",
                mechanism_question="Why doesn't the swing just stop in the middle? What pulls it back?",
                common_misconception="heavier objects swing faster",
                misconception_correction="Weight doesn't affect the period of oscillation. What matters is the length and gravity.",
                real_life_fact="Pendulums were once the world's most precise clocks—every swing marked a second.",
                transfer_question="If we took the swing to a place with no gravity, would it still oscillate?",
                creative_task="Draw a simple pendulum and label the forces acting on it.",
                misconceptions_regex=[r"heavier.*faster", r"weight.*speed", r"mass.*quick"]
            )
        }

    def start_lesson(self, concept_name: str) -> str:
        """Start a new lesson on the given concept"""
        if concept_name not in self.concept_packages:
            return f"Sorry, I don't have a lesson package for '{concept_name}'. Available concepts: {list(self.concept_packages.keys())}"

        # Reset session
        self.controller.reset_session()

        # Get concept package
        concept_pkg = self.concept_packages[concept_name]

        # Start with greeting
        return self.controller.process_message("", concept_pkg)

    def continue_lesson(self, user_message: str, concept_name: str) -> str:
        """Continue the lesson with user input"""
        if concept_name not in self.concept_packages:
            return f"Invalid concept: {concept_name}"

        concept_pkg = self.concept_packages[concept_name]
        return self.controller.process_message(user_message, concept_pkg)

    def get_session_progress(self) -> Dict[str, Any]:
        """Get the current session progress"""
        return {
            "current_state": self.controller.state.current_state.value,
            "retrieval_score": self.controller.state.retrieval_score,
            "transfer_success": self.controller.state.transfer_success,
            "misconceptions_handled": self.controller.state.misconception_detected
        }

def get_api_key():
    """Get API key from environment variable with proper error handling"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables!")
        st.info("Please make sure you have a .env file with your API key:")
        st.code("GROQ_API_KEY=your_api_key_here")
        st.stop()
    return api_key

# Streamlit App
def main():
    st.set_page_config(
        page_title="Adaptive Teaching Agent",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎓 Adaptive Teaching Agent")
    st.markdown("*Powered by Groq API with Llama-4-Maverick*")

    # Get API key with error handling
    try:
        groq_api_key = get_api_key()
    except Exception as e:
        st.error(f"Configuration error: {e}")
        return

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Status
        if groq_api_key:
            st.success("✅ API Key Loaded")
        else:
            st.error("❌ API Key Missing")

        # Concept selection
        st.header("Lesson Selection")
        concept_name = st.selectbox(
            "Choose a concept to learn:",
            ["oscillatory_motion"],
            format_func=lambda x: x.replace("_", " ").title()
        )

        # Session controls
        st.header("Session Controls")
        if st.button("🚀 Start New Lesson"):
            st.session_state.lesson_started = True
            st.session_state.conversation_history = []
            st.session_state.agent = AdaptiveTeachingAgent(groq_api_key)
            initial_response = st.session_state.agent.start_lesson(concept_name)
            st.session_state.conversation_history.append(("Agent", initial_response))
            st.rerun()

        if st.button("🔄 Reset Session"):
            st.session_state.lesson_started = False
            st.session_state.conversation_history = []
            if 'agent' in st.session_state:
                st.session_state.agent.controller.reset_session()
            st.rerun()

        # Debug information
        with st.expander("🔧 Debug Info"):
            st.write(f"API Key Length: {len(groq_api_key) if groq_api_key else 0}")
            st.write(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")

    # Initialize session state
    if 'lesson_started' not in st.session_state:
        st.session_state.lesson_started = False
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Initialize agent only when needed and API key is available
    if 'agent' not in st.session_state and groq_api_key:
        st.session_state.agent = AdaptiveTeachingAgent(groq_api_key)

    # Main content area
    if not st.session_state.lesson_started:
        st.info("👈 Please start a new lesson from the sidebar to begin learning!")
        
        # Show concept information
        st.header("Available Concepts")
        
        with st.expander("📚 Oscillatory Motion"):
            st.markdown("""
            **What you'll learn:**
            - Definition of oscillatory motion
            - Understanding the mechanism behind oscillations
            - Real-world applications of oscillatory motion
            - Critical thinking about motion in different environments
            
            **Teaching approach:**
            - Interactive questioning
            - Misconception handling
            - Real-life context
            - Knowledge retrieval and transfer
            """)
    
    else:
        # Main conversation interface
        st.header("💬 Conversation")
        chat_container = st.container()
        with chat_container:
            for speaker, message in st.session_state.conversation_history:
                if speaker == "Agent":
                    st.markdown(f"🤖 **Agent:** {message}")
                else:
                    st.markdown(f"👨‍🎓 **You:** {message}")
                st.markdown("---")

        # User input
        user_input = st.text_input(
            "Your response:",
            key="user_input",
            placeholder="Type your response here..."
        )

        col_send, col_clear = st.columns([1, 1])
        with col_send:
            if st.button("💬 Send", use_container_width=True):
                if user_input.strip():
                    # Add user message to history
                    st.session_state.conversation_history.append(("You", user_input))
                    # Get agent response
                    try:
                        with st.spinner("Agent is thinking..."):
                            agent_response = st.session_state.agent.continue_lesson(user_input, concept_name)
                        # Add agent response to history
                        st.session_state.conversation_history.append(("Agent", agent_response))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        # Add error to conversation for debugging
                        st.session_state.conversation_history.append(("System", f"Error occurred: {str(e)}"))
                else:
                    st.warning("Please enter a response!")

        with col_clear:
            if st.button("🗑️ Clear Input", use_container_width=True):
                st.session_state.user_input = ""
                st.rerun()

        # Progress section below conversation
        st.markdown("\n---\n")
        st.header("📊 Progress")
        if 'agent' in st.session_state:
            try:
                progress = st.session_state.agent.get_session_progress()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current State", progress['current_state'])
                with col2:
                    st.metric("Retrieval Score", f"{progress['retrieval_score']:.1f}")
                with col3:
                    st.metric("Transfer Success", "✅" if progress['transfer_success'] else "❌")
                with col4:
                    st.metric("Misconceptions", "✅" if progress['misconceptions_handled'] else "❌")
            except Exception as e:
                st.error(f"Could not load progress: {e}")

    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit and Groq API | "
        "🔒 API Key loaded from environment"
    )

if __name__ == "__main__":
    main()