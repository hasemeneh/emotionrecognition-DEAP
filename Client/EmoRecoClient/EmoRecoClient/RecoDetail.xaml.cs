using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace EmoRecoClient
{
	/// <summary>
	/// Interaction logic for RecoDetail.xaml
	/// </summary>
	public partial class RecoDetail : Window
	{
		IndividualModel data = null;
		public RecoDetail(IndividualModel data)
		{
			InitializeComponent();
			this.data = data;
			labelNama.Content = data.Nama;
			if (data.Valence > 0)
			{
				// valence tinggi
				if (data.Arousal > 0)
				{
					// arousal tinggi
					imgActive.Visibility = Visibility.Visible;
					ArousalLabel.Content = "Aktif";
					listPossibleEmotion.Items.Add("Siaga");
					listPossibleEmotion.Items.Add("Girang");
					listPossibleEmotion.Items.Add("Antusias");
					listPossibleEmotion.Items.Add("Gembiar");
					listPossibleEmotion.Items.Add("Bahagia");

				}
				else
				{
					// arousal rendah 
					imgSad.Visibility = Visibility.Visible;
					ArousalLabel.Content = "Tidak Aktif";
					listPossibleEmotion.Items.Add("Puas");
					listPossibleEmotion.Items.Add("Tentram");
					listPossibleEmotion.Items.Add("Santai");
					listPossibleEmotion.Items.Add("Tenang");
				}
				imgSmile.Visibility = Visibility.Visible;
				ValenceLabel.Content = "Menyenangkan";
			}
			else
			{
				// valence rendah
				if (data.Arousal > 0)
				{
					// arousal tinggi
					imgActive.Visibility = Visibility.Visible;
					ArousalLabel.Content = "Aktif";
					listPossibleEmotion.Items.Add("Tegang");
					listPossibleEmotion.Items.Add("Gugup");
					listPossibleEmotion.Items.Add("Stress");
					listPossibleEmotion.Items.Add("Jengkel");
				}
				else
				{
					// low arousal
					imgCalm.Visibility = Visibility.Visible;
					ArousalLabel.Content = "Tidak Aktif";
					listPossibleEmotion.Items.Add("Sedih");
					listPossibleEmotion.Items.Add("Murung");
					listPossibleEmotion.Items.Add("Lesu");
					listPossibleEmotion.Items.Add("Bosan");
				}
				imgSad.Visibility = Visibility.Visible;
				ValenceLabel.Content = "Tidak Menyenangkan";
			}
		}
	}
}
