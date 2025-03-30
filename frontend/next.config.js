/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  reactStrictMode: true,
  swcMinify: true,
  // Explicitly configure the path aliases
  // experimental: {
  //   esmExternals: 'loose',
  // },
};

module.exports = nextConfig;
