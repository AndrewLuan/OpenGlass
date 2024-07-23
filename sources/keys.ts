export const keys = {
    groq: process.env.EXPO_PUBLIC_GROQ_API_KEY ?? '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',
    ollama: process.env.EXPO_PUBLIC_OLLAMA_API_URL ?? 'http://localhost:11434/api/chat',
    openai: process.env.EXPO_PUBLIC_OPENAI_API_KEY ?? 'sk-DXVsSf9w131mIhg4LLG6tY0n0dlsotDPTlWcPRiZJS59RwXq',
};//disable openai if needed