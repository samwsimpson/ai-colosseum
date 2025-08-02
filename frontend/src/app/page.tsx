'use client';

import React from 'react';
import Link from 'next/link';
import { FeatureCard } from '@/components/FeatureCard'; // CORRECTED IMPORT PATH

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

      {/* How It Works Section */}
      <div className="bg-gray-900 py-16">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-4xl font-bold text-white mb-12">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="flex flex-col items-center">
              <FaRobot className="text-blue-500 text-5xl mb-4" />
              <h3 className="text-xl font-bold text-white mb-2">1. The Orchestrator</h3>
              <p className="text-gray-400">A lead AI (ChatGPT) analyzes your request and formulates a plan.</p>
            </div>
            <div className="flex flex-col items-center">
              <FaSync className="text-blue-500 text-5xl mb-4" />
              <h3 className="text-xl font-bold text-white mb-2">2. Intelligent Delegation</h3>
              <p className="text-gray-400">The Orchestrator assigns specific tasks to specialized AIs, like Claude for creative writing.</p>
            </div>
            <div className="flex flex-col items-center">
              <FaLightbulb className="text-blue-500 text-5xl mb-4" />
              <h3 className="text-xl font-bold text-white mb-2">3. Collaborative Output</h3>
              <p className="text-gray-400">The AIs work together, and the Orchestrator synthesizes their responses into a single, comprehensive answer.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Key Features Section */}
      <div className="py-16">
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
              title="Seamless Collaboration"
              description="Witness a natural, conversational workflow as AI agents delegate tasks and work together to deliver the best results."
            />
            <FeatureCard 
              icon={<FaRobot className="text-4xl" />}
              title="Intelligent Orchestration"
              description="Our custom orchestrator AI intelligently manages complex tasks, ensuring every part of your request is handled by the right expert."
            />
          </div>
        </div>
      </div>
    </div>
  );
}