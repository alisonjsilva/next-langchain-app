import { StreamingTextResponse, LangChainStream, Message, OpenAIStream } from 'ai'
import { CallbackManager } from 'langchain/callbacks'
import { ChatOpenAI } from 'langchain/chat_models/openai'
import { AIChatMessage, HumanChatMessage } from 'langchain/schema'
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { HNSWLib } from 'langchain/vectorstores';
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { PineconeClient, ScoredVector } from "@pinecone-database/pinecone";
import { VectorDBQAChain, SimpleSequentialChain } from "langchain/chains";
import { PromptTemplate } from "langchain/prompts";
import { OpenAI } from "langchain/llms/openai";
import { BufferMemory } from "langchain/memory";
import { ConversationChain, LLMChain } from "langchain/chains";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { VectorStoreRetrieverMemory } from "langchain/memory";

export const runtime = 'edge'

export async function POST(req: Request) {
  const { messages } = await req.json()

  const { stream, handlers } = LangChainStream()

  const client = new PineconeClient();
  await client.init({
    environment: process.env.PINECONE_ENVIRONMENT!,
    apiKey: process.env.PINECONE_API_KEY!,
  });

  const pineconeIndex = client.Index(process.env.PINECONE_INDEX as string);

  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings(),
    { pineconeIndex }
  );

  // OpenAI
  const model = new OpenAI({
    streaming: true,
    callbacks: CallbackManager.fromHandlers(handlers),
    modelName: "gpt-3.5-turbo",
  })

  const memoryVectorStore = new MemoryVectorStore(new OpenAIEmbeddings());
  const memory = new VectorStoreRetrieverMemory({
    // 1 is how many documents to return, you might want to return more, eg. 4
    vectorStoreRetriever: vectorStore.asRetriever(1),
    memoryKey: "history",
  });

  // const prompt = new PromptTemplate({
  //   template: "Q: {query}",
  //   inputVariables: ["query"],
  // });

  // const initialChain = new LLMChain({
  //   llm: model,
  //   prompt: reviewPromptTemplate,
  // });

  // set memory
  // const memory = new BufferMemory();

  // const chain = new ConversationChain({ llm: model, prompt, memory: memory });

  const chain1 = VectorDBQAChain.fromLLM(model, vectorStore);

  const overallChain = new SimpleSequentialChain({
    chains: [chain1],
    verbose: true,
    memory: memory,
  });


  const query = (messages as Message[]).map(m =>
    m.role == 'user'
      ? m.content
      : ""
  )?.at(-1);



  overallChain.run(query)


  return new StreamingTextResponse(stream)
}
