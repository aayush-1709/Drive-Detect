import { motion } from "framer-motion";
import { Cpu, Layers, Activity, Database, Zap } from "lucide-react";
import { Navbar } from '../components/layout/Navbar';

const NeuralArchitecture = () => {
  return (
  <>
    <Navbar />

    <div className="min-h-screen bg-white dark:bg-[#020202] text-gray-900 dark:text-white transition-colors duration-300 px-6 py-24">
      <div className="max-w-6xl mx-auto">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-20"
        >
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            Neural <span className="text-blue-600">Architecture</span>
          </h1>
          <p className="text-gray-600 dark:text-gray-400 max-w-3xl mx-auto text-lg leading-relaxed">
            DriveDetect uses a ResNet-50 based Convolutional Neural Network
            optimized for real-time traffic sign recognition with low latency
            inference and high classification accuracy.
          </p>
        </motion.div>

        {/* Architecture Pipeline */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mb-24"
        >
          <h2 className="text-2xl font-semibold text-center mb-12 text-blue-600">
            End-to-End Processing Pipeline
          </h2>

          <div className="flex flex-col md:flex-row items-center justify-between gap-6 text-center">
            {[
              "Input Frame",
              "Convolution Blocks",
              "Feature Extraction",
              "Pooling",
              "Dense Layer",
              "Traffic Sign Prediction"
            ].map((step, index) => (
              <div key={index} className="flex items-center">
                <div className="px-6 py-4 rounded-xl bg-gray-100 dark:bg-white/5 border border-gray-200 dark:border-white/10 shadow-md hover:shadow-blue-500/20 transition-all duration-300">
                  <p className="font-medium">{step}</p>
                </div>

                {index < 5 && (
                  <div className="hidden md:block w-10 h-[2px] bg-blue-500 mx-2" />
                )}
              </div>
            ))}
          </div>
        </motion.div>

        {/* Architecture Cards */}
        <div className="grid md:grid-cols-2 gap-10">

          {/* Backbone */}
          <motion.div
            whileHover={{ y: -6 }}
            className="p-8 rounded-2xl bg-gray-100 dark:bg-white/5 border border-gray-200 dark:border-white/10 shadow-xl transition"
          >
            <div className="flex items-center gap-3 mb-4 text-blue-600">
              <Cpu size={24} />
              <h3 className="text-xl font-semibold">Backbone Network</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-400 leading-relaxed">
              DriveDetect leverages a ResNet-50 backbone for deep feature
              extraction. Residual connections allow efficient gradient flow,
              enabling stable training and improved performance across
              diverse traffic environments.
            </p>
          </motion.div>

          {/* Feature Learning */}
          <motion.div
            whileHover={{ y: -6 }}
            className="p-8 rounded-2xl bg-gray-100 dark:bg-white/5 border border-gray-200 dark:border-white/10 shadow-xl transition"
          >
            <div className="flex items-center gap-3 mb-4 text-blue-600">
              <Layers size={24} />
              <h3 className="text-xl font-semibold">Feature Learning</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-400 leading-relaxed">
              Multiple convolutional blocks extract spatial patterns such as
              shapes, edges, and color distributions from traffic sign images.
              ReLU activation ensures non-linearity while pooling reduces
              spatial dimensions for efficient computation.
            </p>
          </motion.div>

          {/* Optimization */}
          <motion.div
            whileHover={{ y: -6 }}
            className="p-8 rounded-2xl bg-gray-100 dark:bg-white/5 border border-gray-200 dark:border-white/10 shadow-xl transition"
          >
            <div className="flex items-center gap-3 mb-4 text-blue-600">
              <Activity size={24} />
              <h3 className="text-xl font-semibold">Training & Optimization</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-400 leading-relaxed">
              The model is trained using supervised learning on labeled traffic
              sign datasets. Cross-entropy loss and gradient descent
              optimization techniques are applied to maximize classification
              accuracy while maintaining inference stability.
            </p>
          </motion.div>

          {/* Dataset */}
          <motion.div
            whileHover={{ y: -6 }}
            className="p-8 rounded-2xl bg-gray-100 dark:bg-white/5 border border-gray-200 dark:border-white/10 shadow-xl transition"
          >
            <div className="flex items-center gap-3 mb-4 text-blue-600">
              <Database size={24} />
              <h3 className="text-xl font-semibold">Dataset & Classes</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-400 leading-relaxed">
              The training dataset includes regulatory, warning, and mandatory
              traffic sign categories. Data preprocessing techniques such as
              normalization and augmentation improve robustness in varying
              lighting and weather conditions.
            </p>
          </motion.div>

          {/* Real-Time Inference */}
          <motion.div
            whileHover={{ y: -6 }}
            className="md:col-span-2 p-8 rounded-2xl bg-gray-100 dark:bg-white/5 border border-gray-200 dark:border-white/10 shadow-xl transition"
          >
            <div className="flex items-center gap-3 mb-4 text-blue-600">
              <Zap size={24} />
              <h3 className="text-xl font-semibold">Real-Time Inference</h3>
            </div>
            <p className="text-gray-700 dark:text-gray-400 leading-relaxed">
              The architecture is optimized for low-latency inference,
              enabling detection within milliseconds. This ensures DriveDetect
              can process live video streams efficiently without compromising
              classification accuracy.
            </p>
          </motion.div>

        </div>
      </div>
    </div>
  </>
);
};

export default NeuralArchitecture;