# recover-noise-or-unclear-img
ノイズ・虫食い・粗さのある画像を綺麗にするWEBアプリ

<h2>アプリの使い方</h2>
起動URL：
<br>
https://pa-up-recover-noise-or-unclear-img-main-zyad3k.streamlit.app/
<br>
<br>
WEBアプリにアクセスすると、
<ul>
<li>ボケやモザイクなど粗さのある虫食い画像</li>
<li>ノイズのある画像</li>
</ul>
などを綺麗にできるページへアクセスできます


<h2>ソースコード・環境</h2>
本WEBアプリでは機械学習モデルCAEを用いて、ノイズ除去や超解像を行い、上述のような画像を綺麗に補正します。
CAEにスキップコネクションを実装することで、ボケや粗さの少ない画像への復元を可能としています。
<br>
ImageNet2010のValidation画像の中から、4900枚の学習画像、700枚の検証画像をランダムに選定して、CAEモデルを学習させました。

