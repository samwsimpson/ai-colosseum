'use client';
// test
import React from 'react';
import Link from 'next/link';
import { FeatureCard } from '@/components/FeatureCard';
import { ModelCard } from '@/components/ModelCard';

// You will need to install react-icons to use these. Run: npm install react-icons
import { FaRobot, FaSync, FaLightbulb, FaTools } from 'react-icons/fa';

export default function HomePage() {
  return (
    <div className="flex-grow">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16 flex flex-col items-center justify-center text-center">
        <h2 className="text-5xl md:text-7xl font-extrabold text-white mb-4">
          Unleash the Power of Collaborative AI
        </h2>
        <p className="text-xl text-gray-400 mb-8 max-w-3xl">
          The Colosseum is a groundbreaking platform where multiple AI models work together to solve complex problems, generate creative content, and bring your ideas to life.
        </p>
        <Link href="/chat">
          <button className="bg-blue-600 text-white text-lg font-bold py-3 px-8 rounded-full hover:bg-blue-700 transition duration-300 shadow-lg">
            Get Started
          </button>
        </Link>
      </div>

      {/* NEW: Updated How It Works Section */}
      <div className="bg-gray-900 py-16">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-4xl font-bold text-white mb-12">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="flex flex-col items-center">
              <FaRobot className="text-blue-500 text-5xl mb-4" />
              <h3 className="text-xl font-bold text-white mb-2">1. The Prompt</h3>
              <p className="text-gray-400">You send a request to the AI agents and start a group conversation.</p>
            </div>
            <div className="flex flex-col items-center">
              <FaSync className="text-blue-500 text-5xl mb-4" />
              <h3 className="text-xl font-bold text-white mb-2">2. Dynamic Collaboration</h3>
              <p className="text-gray-400">AI agents dynamically collaborate, passing control to each other as needed to complete the task.</p>
            </div>
            <div className="flex flex-col items-center">
              <FaLightbulb className="text-blue-500 text-5xl mb-4" />
              <h3 className="text-xl font-bold text-white mb-2">3. The Solution</h3>
              <p className="text-gray-400">Each agent contributes, and the final response is a product of their seamless group effort.</p>
            </div>
          </div>
        </div>
      </div>

      {/* AI Models Section */}
      <div className="py-16 bg-gray-950">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-4xl font-bold text-white mb-12">Our AI Models</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
            <ModelCard
              logoSrc="/logos/openai.svg"
              name="ChatGPT"
              description="A powerful language model for natural language processing and generation."
            />
            <ModelCard
              logoSrc="/logos/anthropic.svg"
              name="Claude"
              description="A large language model focused on safety, helpfulness, and harmlessness."
            />
            <ModelCard
              logoSrc="/logos/gemini.png"
              name="Gemini"
              description="A family of multimodal models optimized for performance and versatility."
            />
            <ModelCard
              logoSrc="/logos/mistral.svg"
              name="Mistral"
              description="An efficient and powerful open-source large language model for a wide range of tasks."
            />
          </div>
        </div>
      </div>

      {/* NEW: Updated Key Features Section */}
      <div className="py-16 bg-gray-900">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-4xl font-bold text-white mb-12">Key Features</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <FeatureCard 
              icon={<FaTools className="text-4xl" />}
              title="Multi-Model Integration"
              description="Connect and use a wide range of state-of-the-art AI models, from large language models to specialized image and data AIs."
            />
            <FeatureCard 
              icon={<FaSync className="text-4xl" />}
              title="Autonomous Collaboration"
              description="Witness a natural, conversational workflow as AI agents delegate tasks and work together to deliver the best results."
            />
            <FeatureCard 
              icon={<FaRobot className="text-4xl" />}
              title="GroupChat Automation"
              description="Our custom GroupChat manager intelligently handles complex tasks, ensuring every part of your request is handled by the right expert."
            />
          </div>
        </div>
      </div>
    </div>
  );
}
