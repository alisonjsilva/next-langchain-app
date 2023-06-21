import { StreamingTextResponse, LangChainStream, Message, OpenAIStream } from 'ai'
import { CallbackManager } from 'langchain/callbacks'
import { ChatOpenAI } from 'langchain/chat_models/openai'
import { AIChatMessage, HumanChatMessage } from 'langchain/schema'
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { HNSWLib } from 'langchain/vectorstores';
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { PineconeClient, ScoredVector } from "@pinecone-database/pinecone";
import { VectorDBQAChain } from "langchain/chains";
import { PromptTemplate } from "langchain/prompts";

export const runtime = 'edge'



export async function POST(req: Request) {
  const { messages } = await req.json()
  // const vectorStore = await HNSWLib.load("Awards_Ratings__TO_IMPORT.index", new OpenAIEmbeddings());
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

  // const results = await vectorStore.similaritySearch(
  //   "How many Awards?",
  // );
  // console.log(results);
  let streamedResponse = "";
  const { stream, handlers } = LangChainStream()

  const openai = new ChatOpenAI({
    streaming: true,
    callbackManager: CallbackManager.fromHandlers(handlers),
    modelName: "gpt-3.5-turbo",
  })

  const query = (messages as Message[]).map(m =>
    m.role == 'user'
      ? m.content
      : ""
  )?.at(-1);

  const chain = VectorDBQAChain.fromLLM(openai, vectorStore);
  // const chainResult = await chain.call({ query: query });

  chain.run(query)

  // llmStreamModel
  //   .call(
  //     (messages as Message[]).map(m =>
  //       m.role == 'user'
  //         ? new HumanChatMessage(m.content)
  //         : new AIChatMessage(m.content)
  //     )
  //   )
  //   .catch(console.error)

  return new StreamingTextResponse(stream)
}
