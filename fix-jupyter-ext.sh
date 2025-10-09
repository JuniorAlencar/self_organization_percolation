# 1) Feche o VS Code
pkill -f code || true

# 2) Mostre o que ainda existe
ls -al ~/.vscode/extensions | grep -i jupyter || true
ls -al ~/snap/code/common/.vscode/extensions | grep -i jupyter || true

# 3) Remova forçando (usa sudo só para APAGAR pastas travadas por root)
sudo rm -rf \
  ~/.vscode/extensions/ms-toolsai.jupyter* \
  ~/.vscode/extensions/ms-toolsai.jupyter-renderers* \
  ~/snap/code/common/.vscode/extensions/ms-toolsai.jupyter* \
  ~/snap/code/common/.vscode/extensions/ms-toolsai.jupyter-renderers*

# 4) Garanta dono correto das pastas
sudo chown -R "$USER":"$USER" ~/.vscode ~/.config/Code ~/snap/code 2>/dev/null || true

# 5) Apague caches que podem re-referenciar a versão antiga
rm -rf ~/.config/Code/User/globalStorage/ms-toolsai.jupyter \
       ~/.config/Code/CachedExtensionVSIXs || true

# 6) Confirme que sumiu de vez
find ~/.vscode -maxdepth 3 -iname "ms-toolsai.jupyter*" -print
find ~/snap/code/common/.vscode -maxdepth 3 -iname "ms-toolsai.jupyter*" -print

