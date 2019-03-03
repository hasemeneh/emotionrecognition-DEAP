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
				if (data.Arousal > 0)
				{
					imgActive.Visibility = Visibility.Visible;
					ArousalLabel.Content = "Active";
					listPossibleEmotion.Items.Add("Alert");
					listPossibleEmotion.Items.Add("Excited");
					listPossibleEmotion.Items.Add("Enthusiastic");
					listPossibleEmotion.Items.Add("Elated");
					listPossibleEmotion.Items.Add("Happy");

				}
				else
				{
					imgSad.Visibility = Visibility.Visible;
					ArousalLabel.Content = "Inactive";
					listPossibleEmotion.Items.Add("Contended");
					listPossibleEmotion.Items.Add("Serene");
					listPossibleEmotion.Items.Add("Relaxed");
					listPossibleEmotion.Items.Add("Calm");
				}
				imgSmile.Visibility = Visibility.Visible;
				ValenceLabel.Content = "Pleasant";
			}
			else
			{
				if (data.Arousal > 0)
				{
					imgActive.Visibility = Visibility.Visible;
					ArousalLabel.Content = "Active";
					listPossibleEmotion.Items.Add("Tense");
					listPossibleEmotion.Items.Add("Nervous");
					listPossibleEmotion.Items.Add("Stressed");
					listPossibleEmotion.Items.Add("Upset");
				}
				else
				{
					imgCalm.Visibility = Visibility.Visible;
					ArousalLabel.Content = "Inactive";
					listPossibleEmotion.Items.Add("Sad");
					listPossibleEmotion.Items.Add("Depressed");
					listPossibleEmotion.Items.Add("Sluggish");
					listPossibleEmotion.Items.Add("Bored");
				}
				imgSad.Visibility = Visibility.Visible;
				ValenceLabel.Content = "Unpleasant";
			}
		}
	}
}
