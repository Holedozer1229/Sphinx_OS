/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  async rewrites() {
    const excaliburOracleUrl =
      process.env.NEXT_PUBLIC_EXCALIBUR_ORACLE_URL ||
      'https://oracle.excaliburcrypto.com'
    return [
      {
        source: '/excalibur/oracle/:path*',
        destination: `${excaliburOracleUrl}/:path*`,
      },
      {
        source: '/web/knights-round-table/:path*',
        destination: 'https://www.excaliburcrypto.com/web/knights-round-table/:path*',
      },
    ]
  },
}

module.exports = nextConfig
