// frontend/src/components/ModelCard.tsx
import React from 'react';
import Image from 'next/image';

interface ModelCardProps {
  logoSrc: string;
  name: string;
  description: string;
}

export const ModelCard = ({ logoSrc, name, description }: ModelCardProps) => {
  return (
    <div className="bg-gray-900 rounded-3xl p-6 shadow-2xl transition-all duration-300 ease-in-out transform hover:scale-105 hover:shadow-blue-500/50">
      <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
        {/* Using Next.js Image component for optimization */}
        <div className="relative w-32 h-32 mb-4">
          <Image
            src={logoSrc}
            alt={`${name} logo`}
            layout="fill"
            objectFit="contain"
            className="transition-transform duration-300 transform group-hover:scale-110"
          />
        </div>
        <h3 className="text-2xl font-bold text-white">{name}</h3>
        <p className="text-gray-400 text-sm">{description}</p>
      </div>
    </div>
  );
};
