import { default as CopyPlugin } from 'copy-webpack-plugin';
import { default as HtmlWebpackPlugin } from 'html-webpack-plugin';
import { default as MiniCssExtractPlugin } from 'mini-css-extract-plugin';

const isDev = process.env.NODE_ENV !== 'production';
const mode = isDev ? 'development' : 'production';

export default [
  {
    mode,
    entry: './index.mjs',
    target: 'web',
    output: {
      filename: 'app.js',
      path: new URL('./dist', import.meta.url).pathname,
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
      ],
    },
    plugins: [
      new HtmlWebpackPlugin({
        title: 'jxl-oxide-wasm Demo',
      }),
      new CopyPlugin({
        patterns: [
          { from: 'assets/', to: 'assets/' },
        ],
      }),
      ...(isDev ? [] : [new MiniCssExtractPlugin()]),
    ],
  },
  {
    mode,
    entry: './service-worker.js',
    target: 'webworker',
    output: {
      filename: 'service-worker.js',
      path: new URL('./dist', import.meta.url).pathname,
    },
    experiments: {
      asyncWebAssembly: true,
    },
  },
];
