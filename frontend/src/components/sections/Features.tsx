import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Scan, Zap, ShieldCheck, Lock, Cpu, Server, Play } from 'lucide-react';

export const Features = () => {
  const [latencyState, setLatencyState] = useState<'processing' | 'result'>('processing');
  const [msCount, setMsCount] = useState(0);

  useEffect(() => {
    const cycleAnimation = () => {
      setLatencyState('processing');
      setMsCount(0);

      setTimeout(() => {
        setLatencyState('result');
        let start = 0;
        const interval = setInterval(() => {
          start += 1;
          setMsCount(start);
          if (start >= 12) clearInterval(interval);
        }, 50);
      }, 2000);
    };

    cycleAnimation();
    const loop = setInterval(cycleAnimation, 4000);
    return () => clearInterval(loop);
  }, []);

  return (
    <section
      id="features"
      className="relative py-32 border-t border-black/10 dark:border-white/5"
    >
      {/* Background Glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 w-[900px] h-[900px] -translate-x-1/2 -translate-y-1/2 bg-blue-600/5 blur-[120px] rounded-full"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <h2 className="text-3xl md:text-5xl font-bold text-gray-900 dark:text-white mb-6">
            System{' '}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-400">
              Capabilities
            </span>
          </h2>
          <p className="text-gray-600 dark:text-gray-400 max-w-2xl mx-auto text-lg leading-relaxed">
            Designed for real-time traffic sign recognition with high accuracy,
            low latency inference, and secure processing pipelines.
          </p>
        </motion.div>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">

          {/* Detection Engine */}
          <motion.div
            whileHover={{ y: -6 }}
className="md:col-span-2 rounded-3xl border border-black/10 dark:border-white/10 bg-white/40 dark:bg-white/5 backdrop-blur-md p-8 shadow-sm hover:shadow-xl transition-all duration-300 flex flex-col"          >
            <div className="flex items-center gap-3 mb-5">
              <div className="p-3 rounded-xl bg-blue-500/20 text-blue-500">
                <Scan size={24} />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                Detection Engine
              </h3>
            </div>

            <p className="text-gray-700 dark:text-gray-300 mb-6 leading-relaxed">
              Powered by a deep Convolutional Neural Network architecture
              optimized for identifying regulatory, warning, and mandatory
              traffic signs across varying weather and lighting conditions.
            </p>

            <div className="flex-1 rounded-xl bg-gray-50 dark:bg-black/40 border border-gray-200 dark:border-white/10 relative overflow-hidden p-6">

           <div className="grid grid-cols-3 gap-4 mb-6">
                {["Feature Map", "Edge Detection", "Pattern Match"].map((label, i) => (
                  <div
                    key={i}
                    className="h-16 rounded-lg bg-white dark:bg-white/5 border border-gray-300 dark:border-white/10 flex items-center justify-center text-xs font-medium text-gray-700 dark:text-gray-200 shadow-sm"
                  >
                    {label}
                  </div>
                ))}
              </div>
              
              <div className="h-16 rounded-lg bg-blue-600 dark:bg-blue-500/20 border border-blue-500/60 flex items-center justify-center shadow-sm">
                <span className="text-sm font-semibold text-white dark:text-blue-200 tracking-wider">
                  OBJECT DETECTED
                </span>
              </div>
            </div>
          </motion.div>

          {/* Low Latency */}
          <motion.div
            whileHover={{ y: -6 }}
            className="rounded-3xl border border-black/10 dark:border-white/10 bg-white/40 dark:bg-white/5 backdrop-blur-md p-8 shadow-sm hover:shadow-xl transition-all duration-300 flex flex-col"
          >
            <div className="flex items-center gap-3 mb-5">
              <div className="p-3 rounded-xl bg-green-500/20 text-green-500">
                <Zap size={24} />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                Low Latency
              </h3>
            </div>

            <div className="flex-1 flex items-center justify-center">
              {latencyState === 'processing' ? (
                <div className="flex items-center gap-3">
                  {[Cpu, Server, Play].map((Icon, i) => (
                    <motion.div
                      key={i}
                      animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
                      transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.2 }}
                    >
                      <Icon className="text-green-500" size={24} />
                    </motion.div>
                  ))}
                </div>
              ) : (
                <div className="text-center">
                  <span className="text-6xl font-mono font-bold text-gray-900 dark:text-white">
                    {msCount}
                    <span className="text-2xl text-green-500 ml-1">ms</span>
                  </span>
                  <p className="text-sm text-gray-500 mt-2 uppercase tracking-widest font-mono">
                    Inference Time
                  </p>
                </div>
              )}
            </div>
          </motion.div>

          {/* Neural Network Architecture */}
          <motion.div
            whileHover={{ y: -6 }}
            className="rounded-3xl border border-black/10 dark:border-white/10 bg-white/40 dark:bg-white/5 backdrop-blur-md p-8 shadow-sm hover:shadow-xl transition-all duration-300"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="p-3 rounded-xl bg-purple-500/20 text-purple-500">
                <Cpu size={24} />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                Neural Architecture
              </h3>
            </div>

            <div className="space-y-4 font-mono text-sm text-gray-600 dark:text-gray-400">
              <div className="flex justify-between border-b border-black/10 dark:border-white/5 pb-2">
                <span>Model</span>
                <span className="text-gray-900 dark:text-white">ResNet-50</span>
              </div>
              <div className="flex justify-between border-b border-black/10 dark:border-white/5 pb-2">
                <span>Parameters</span>
                <span className="text-gray-900 dark:text-white">25.6M</span>
              </div>
              <div className="flex justify-between border-b border-black/10 dark:border-white/5 pb-2">
                <span>Accuracy</span>
                <span className="text-purple-500">99.8%</span>
              </div>
            </div>
          </motion.div>

          {/* Secure Pipeline */}
          <motion.div
            whileHover={{ y: -6 }}
            className="md:col-span-2 rounded-3xl border border-black/10 dark:border-white/10 bg-white/40 dark:bg-white/5 backdrop-blur-md p-8 shadow-sm hover:shadow-xl transition-all duration-300"
          >
            <div className="flex flex-col sm:flex-row items-center justify-between gap-8">
              <div className="max-w-sm">
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-3 rounded-xl bg-cyan-500/20 text-cyan-500">
                    <ShieldCheck size={24} />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                    Secure & Encrypted Pipeline
                  </h3>
                </div>
                <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
                  Ensures integrity across preprocessing, inference, and output
                  layers with secured data handling to prevent adversarial
                  manipulation or data corruption.
                </p>
              </div>

              <div className="h-32 w-48 flex items-center justify-center bg-black/5 dark:bg-black/40 rounded-xl border border-black/10 dark:border-white/5">
                <motion.div
                  animate={{ rotateY: [0, 180, 360] }}
                  transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
                >
                  <Lock className="text-cyan-500 w-12 h-12" />
                </motion.div>
              </div>
            </div>
          </motion.div>

        </div>
      </div>
    </section>
  );
};