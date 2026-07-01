export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  image_base64?: string;
  annotated_image?: string | null;
}

export interface TokenUsage {
  input: number;
  output: number;
  total: number;
}

export interface ChatResponse {
  response: string;
  prediction_id?: string | null;
  annotated_image?: string | null;
  agent_loop_time_s?: number;
  iterations?: number;
  tools_called?: string[];
  context_limit_exceeded?: boolean;
  tokens_used?: TokenUsage;
}
