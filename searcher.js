require('dotenv').config();
const { OpenAI } = require("openai");
const readlineSync = require('readline-sync');
const fs = require('fs');

// Initialize OpenAI with API key from environment variables
const openai = new OpenAI(process.env.OPENAI_API_KEY);

// Load data from a JSON file
function loadData(filePath) {
  try {
    const rawData = fs.readFileSync(filePath);
    return JSON.parse(rawData);
  } catch (error) {
    console.error('Error reading the JSON file:', error);
    return null;
  }
}

// Save Q&A pairs to a JSON file
function saveQAPairs(filePath, qaPairs) {
  try {
    fs.writeFileSync(filePath, JSON.stringify(qaPairs, null, 2));
    console.log('Q&A pairs saved successfully.');
  } catch (error) {
    console.error('Error writing Q&A pairs to file:', error);
  }
}

// Generate embeddings using OpenAI's model
async function generateEmbedding(text) {
  try {
    const response = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: text
    });
    return response.data.data[0].embedding;
  } catch (error) {
    console.error('Error generating embedding:', error);
    return null;
  }
}

// Calculate cosine similarity between two vectors
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0.0;
  let normA = 0.0;
  let normB = 0.0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  if (normA === 0 || normB === 0) {
    return 0; // Avoid division by zero
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Perform semantic search to find the best matching content
async function semanticSearch(queryEmbedding, data) {
  let bestMatch = { id: null, score: -1 };

  data.forEach(item => {
    const similarity = cosineSimilarity(queryEmbedding, item.embedding);
    if (similarity > bestMatch.score) {
      bestMatch = { id: item.id, score: similarity };
    }
  });

  return bestMatch;
}

// Generate text using GPT-4 based on a given prompt
async function generateText(content) {
  try {
    const response = await openai.createCompletion({
      model: "gpt-4", // or another model of your choice
      prompt: content,
      max_tokens: 150 // Adjust as needed
    });
    return response.data.choices[0].text.trim();
  } catch (error) {
    console.error('Error generating text:', error);
    return null;
  }
}

// Main function to orchestrate the application logic
async function main() {
  const filePath = 'negativa_structured.json'; // Path to your JSON file
  let data = loadData(filePath); // Load data from JSON file
  let qaPairs = []; // Array to store Q&A pairs

  while (true) {
    console.log("\nOptions:");
    console.log("1: Make a search query");
    console.log("2: Log question and answer");
    console.log("3: Exit");

    const choice = readlineSync.question('Enter your choice: ');

    switch (choice) {
      case '1':
        await handleSearchQuery(data, qaPairs);
        break;
      case '2':
        handleManualLog(qaPairs);
        break;
      case '3':
        console.log('Exiting...');
        return;
      default:
        console.log('Invalid choice. Please enter 1, 2, or 3.');
        break;
    }
  }
}

// Handle search query logic
async function handleSearchQuery(data, qaPairs) {
  const query = readlineSync.question('Enter your search query: ');
  const queryEmbedding = await generateEmbedding(query);
  if (!queryEmbedding) {
    console.log('Failed to generate query embedding.');
    return;
  }

  const bestMatch = await semanticSearch(queryEmbedding, data);
  if (bestMatch && bestMatch.id !== null) {
    const content = data.find(item => item.id === bestMatch.id)?.content;
    if (content) {
      const generatedText = await generateText(content);
      if (generatedText) {
        console.log(`Generated text based on the best match: ${generatedText}`);
        qaPairs.push({ question: query, answer: generatedText });
        saveQAPairs('qa_pairs.json', qaPairs);
      } else {
        console.log('Failed to generate text from the GPT model.');
      }
    } else {
      console.log('Content for the best match not found.');
    }
  } else {
    console.log('No matches found.');
  }
}

// Handle manual logging of Q&A pairs
function handleManualLog(qaPairs) {
  const question = readlineSync.question('Enter a question: ');
  const answer = readlineSync.question('Enter the answer: ');
  qaPairs.push({ question: question, answer: answer });
  saveQAPairs('qa_pairs.json', qaPairs);
}

main();
