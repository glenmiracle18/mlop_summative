/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  // Removed swcMinify as it's no longer needed in Next.js 15
  // Explicitly configure the path aliases
  // experimental: {
  //   esmExternals: 'loose',
  // },
};

module.exports = nextConfig;
