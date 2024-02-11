import { default as HtmlWebpackPlugin } from 'html-webpack-plugin';
import { default as MiniCssExtractPlugin } from 'mini-css-extract-plugin';

const isDev = process.env.NODE_ENV !== 'production';
const mode = isDev ? 'development' : 'production';

const outputPath = new URL('./dist', import.meta.url).pathname;

export default [
  {
    mode,
    entry: './src/index.mjs',
    target: 'web',
    output: {
      filename: 'assets/[name].[contenthash].js',
      chunkFilename: 'assets/[chunkhash].js',
      assetModuleFilename: 'assets/[name].[hash][ext][query]',
      webassemblyModuleFilename: 'assets/[hash].wasm',
      path: outputPath,
    },
    module: {
      rules: [
        {
          test: /\.css$/i,
          use: [
            isDev ? 'style-loader' : MiniCssExtractPlugin.loader,
            'css-loader',
          ],
        },
        {
          test: /\.jxl$/i,
          use: [
            {
              loader: 'file-loader',
              options: {
                name: 'assets/[name].[hash].[ext]',
              },
            },
          ],
        },
        {
          test: /\.html$/i,
          use: ['html-loader'],
        },
      ],
    },
    plugins: [
      new HtmlWebpackPlugin({
        template: './src/index.html',
      }),
      ...(
        isDev
        ? []
        : [new MiniCssExtractPlugin({ filename: 'assets/[chunkhash].css' })]
      ),
    ],
  },
  {
    mode,
    entry: {
      'service-worker': './src/service-worker.mjs',
      'jxl-decode-worker': './src/jxl-decode-worker.mjs',
    },
    target: 'webworker',
    output: {
      filename: '[name].js',
      chunkFilename: 'assets/[chunkhash].js',
      assetModuleFilename: 'assets/[name].[hash][ext][query]',
      webassemblyModuleFilename: 'assets/[hash].wasm',
      path: outputPath,
    },
    experiments: {
      asyncWebAssembly: true,
    },
  },
];
