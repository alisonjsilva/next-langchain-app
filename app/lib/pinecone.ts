import { PineconeStore } from "langchain/vectorstores/pinecone";
import { PineconeClient, ScoredVector } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";

export async function pineconeClient() {
    const client = new PineconeClient();
    await client.init({
        environment: process.env.PINECONE_ENVIRONMENT!,
        apiKey: process.env.PINECONE_API_KEY!,
    });
    return client;
}

export async function pineconeVectorStoreFromIndex(pineconeIndex: any) {
  return await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings(),
    { pineconeIndex }
  );
}