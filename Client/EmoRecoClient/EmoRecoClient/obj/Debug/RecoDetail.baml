   M S B A M L   `   `   `  ����H  DEmoRecoClient, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null-clr-namespace:EmoRecoClientEmoRecoClient  5   	       EmoRecoClient.RecoDetail   5   	   W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    5   	   R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	   @d2http://schemas.microsoft.com/expression/blend/2008     5   	   Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5   	   V RMaterialDesignThemes.Wpf, Version=2.5.0.1205, Culture=neutral, PublicKeyToken=nullFmaterialDesign1http://materialdesigninxaml.net/winfx/xaml/themes  5      'localclr-namespace:EmoRecoClient  5   	     � WindowStartupLocation$  CenterScreen=�5       � Title$ 
RecoDetail��5	   	   $��426.507��6   $��685.237��6-   .��5
      � .+�5      �� $
��Left=�6
   $	��100��6%   $��	10,10,0,0q�62   $��Image/male-circle-512.png��6E   $	��Top=�6h   $	��100��6�   �� 5      $
��Nama��6
   $
��Left=�6   $��
115,10,0,0q�64   $	��Top=�6H   $��18(�6`   #  MaterialDesignThemes.Wpf.Card  5      $��	10,0,0,10q�6   $	��200��6+   $
��Left=�67   $��Bottom=�6R   .��5      �  � RowDefinitions 5      �� 5      $	��140 �6   �� 5      $��* �6   �� 5      $
��Auto �6   .+�5      �� -   5        $��imgSmile��5      $y�	Collapsed=�6   $.��(Image/smile_emoji_emoticon_by_Vexels.png��63   $	��140��6e   $	��196��6r   $��Uniform=�6~   �� 5      -   5        $��imgSad��5      $y�	Collapsed=�6   $1��+Image/emoji-emoticon-sad-face-by-Vexels.png��61   $	��140��6f   $	��196��6s   $��Uniform=�6   �� 5      $��1��6   $��8 24 8 0q�6   .+�5      �� $��16(�6   $
��Bold"�6   	Valence6:   5      �� 5      $��Right=�6   $��2��6-    �� Orientation$ 
Horizontal=�6:   $��8q�6S   .+�5      �� -   5        $��ValenceLabel��5      $��Active��6!   $
��Bold"�62   5      5      5        5      $��
237,0,0,10q�6   $	��200��6,   $
��Left=�68   $��Bottom=�6S   .��5       �  5!      �� 5"      $	��140 �6   �� 5#      $��* �6   �� 5$      $
��Auto �6   .+�5&      �� -   5        $��	imgActive��5&      $y�	Collapsed=�6   $��Image/clipart987144.png��64   $	��140��6U   $	��196��6b   $��Uniform=�6n   �� 5'      -   5        $��imgCalm��5'      $y�	Collapsed=�6   $��Image/meditate.png��62   $	��140��6N   $	��196��6[   $��Uniform=�6g   �� 5(      $��1��6   $��8 24 8 0q�6   .+�5)      �� $��16(�6   $
��Bold"�6   	Arousal6:   5*      �� 5+      $��Right=�6   $��2��6-   $ 
Horizontal=�6:   $��8q�6S   .+�5,      �� -   5        $��ArousalLabel��5,      $��Active��6!   $
��Bold"�62   5-      5.      5/        50      $��
459,0,0,10q�6   $	��200��6,   $
��Left=�68   $��Bottom=�6S   $	��227��6n   .��51      � .+�52      �� $��Kemungkinan emosi :��6   $
��Bold"�6*   $��16(�6<   �� 53      -   5        $��listPossibleEmotion��53      $��0,34,0,0q�6)   54      55      56      �� 57      -   5        $��	labelNama��57   
   $��Rivaldi Ridhla Julviar��6   $
��Left=�6<   $��
177,10,0,0q�6W   $	��Top=�6k   $��18(�6�   58      59      