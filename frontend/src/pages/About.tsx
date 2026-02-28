import React from 'react';
import { Navbar } from '../components/layout/Navbar';
import { Footer } from '../components/layout/Footer';
import { motion } from 'framer-motion';
import { Target, Users, Shield } from 'lucide-react';

export const About = () => {
  return (
    <div className="min-h-screen bg-transparent text-gray-900 dark:text-gray-200">
      <Navbar />

      <main className="pt-24 pb-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">

          {/* Hero Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h1 className="text-4xl sm:text-5xl font-bold mb-6">
              About <span className="text-blue-500">DriveDetect</span>
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              Making traffic sign recognition accessible and reliable for everyone
            </p>
          </motion.div>

          {/* Mission Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mb-16"
          >
            <div className="bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 rounded-2xl p-8">
              <h2 className="text-2xl font-bold mb-6 text-center">
                Our Mission
              </h2>
              <p className="text-lg text-gray-700 dark:text-gray-300 leading-relaxed text-center">
                At DriveDetect, we believe that advanced computer vision technology should be accessible to everyone.
                Our mission is to make traffic sign recognition reliable, fast, and easy to integrate into any application,
                from autonomous vehicles to safety systems and educational tools.
              </p>
            </div>
          </motion.div>

          {/* Values Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16"
          >
            {[
              {
                icon: <Target className="w-8 h-8" />,
                title: 'Accuracy First',
                description:
                  'We prioritize precision in traffic sign detection to ensure safety and reliability in critical applications.',
              },
              {
                icon: <Users className="w-8 h-8" />,
                title: 'Accessibility',
                description:
                  'Making powerful AI technology available to developers, researchers, and organizations of all sizes.',
              },
              {
                icon: <Shield className="w-8 h-8" />,
                title: 'Open Source',
                description:
                  'Committed to transparency and community-driven development for the benefit of all users.',
              },
            ].map((value, index) => (
              <div
                key={index}
                className="bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 rounded-xl p-6 text-center hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
              >
                <div className="text-blue-500 mb-4 flex justify-center">
                  {value.icon}
                </div>
                <h3 className="text-xl font-semibold mb-3">
                  {value.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-400">
                  {value.description}
                </p>
              </div>
            ))}
          </motion.div>

          {/* Technology Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.6 }}
            className="text-center"
          >
            <h2 className="text-2xl font-bold mb-6">
              Powered by Advanced Technology
            </h2>
            <p className="text-gray-600 dark:text-gray-400 mb-8">
              Our system combines state-of-the-art deep learning models with optimized computer vision pipelines
              to deliver real-time traffic sign recognition with industry-leading performance.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
              {[
                { name: 'ResNet-50', desc: 'Deep Learning Architecture' },
                { name: 'OpenCV', desc: 'Computer Vision Pipeline' },
                { name: 'ONNX', desc: 'Cross-Platform Inference' },
              ].map((tech, idx) => (
                <div
                  key={idx}
                  className="bg-black/5 dark:bg-white/5 border border-black/10 dark:border-white/10 rounded-lg p-4"
                >
                  <div className="font-semibold text-blue-500">
                    {tech.name}
                  </div>
                  <div className="text-gray-500 dark:text-gray-400">
                    {tech.desc}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </main>

      <Footer />
    </div>
  );
};
